# tasks/mediaserver_jellyfin.py

import requests
import logging
import os
import config

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300

# ##############################################################################
# JELLYFIN IMPLEMENTATION
# ##############################################################################


def _jellyfin_get_users(token):
    """Fetches a list of all users from Jellyfin using a provided token."""
    url = f"{config.JELLYFIN_URL}/Users"
    headers = {"X-Emby-Token": token}
    try:
        r = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"Jellyfin get_users failed: {e}", exc_info=True)
        return None


def resolve_user(identifier, token):
    """
    Resolves a Jellyfin username to a User ID.
    If the identifier doesn't match any username, it's returned as is, assuming it's already an ID.
    """
    users = _jellyfin_get_users(token)
    if users:
        for user in users:
            if user.get("Name", "").lower() == identifier.lower():
                logger.info(
                    f"Matched username '{identifier}' to User ID '{user['Id']}'."
                )
                return user["Id"]

    logger.info(f"No username match for '{identifier}'. Assuming it is a User ID.")
    return identifier  # Return original identifier if no match is found


# --- ADMIN/GLOBAL JELLYFIN FUNCTIONS ---
def get_recent_albums(limit):
    """
    Fetches a list of the most recently added albums from Jellyfin using pagination.
    Uses global admin credentials.
    """
    all_albums = []
    start_index = 0
    page_size = 500
    fetch_all = limit == 0
    while fetch_all or len(all_albums) < limit:
        size_to_fetch = (
            page_size if fetch_all else min(page_size, limit - len(all_albums))
        )
        if size_to_fetch <= 0:
            break
        url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
        params = {
            "IncludeItemTypes": "MusicAlbum",
            "SortBy": "DateCreated",
            "SortOrder": "Descending",
            "Recursive": True,
            "Limit": size_to_fetch,
            "StartIndex": start_index,
        }
        try:
            r = requests.get(
                url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
            )
            r.raise_for_status()
            response_data = r.json()
            albums = response_data.get("Items", [])
            if not albums:
                break
            all_albums.extend(albums)
            start_index += len(albums)
            if len(albums) < size_to_fetch:
                break
            if fetch_all and start_index >= response_data.get(
                "TotalRecordCount", float("inf")
            ):
                break
        except Exception as e:
            logger.error(f"Jellyfin get_recent_albums failed: {e}", exc_info=True)
            break
    return all_albums


def get_tracks_from_album(album_id):
    """Fetches all audio tracks for a given album ID from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"ParentId": album_id, "IncludeItemTypes": "Audio"}
    try:
        r = requests.get(
            url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(
            f"Jellyfin get_tracks_from_album failed for album {album_id}: {e}",
            exc_info=True,
        )
        return []


def download_track(temp_dir, item):
    """Downloads a single track from Jellyfin using admin credentials."""
    try:
        track_id = item["Id"]
        file_extension = os.path.splitext(item.get("Path", ""))[1] or ".tmp"
        download_url = f"{config.JELLYFIN_URL}/Items/{track_id}/Download"
        local_filename = os.path.join(temp_dir, f"{track_id}{file_extension}")
        with requests.get(
            download_url, headers=config.HEADERS, stream=True, timeout=REQUESTS_TIMEOUT
        ) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded '{item['Name']}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(
            f"Failed to download track {item.get('Name', 'Unknown')}: {e}",
            exc_info=True,
        )
        return None


def get_all_songs():
    """Fetches all songs from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Audio", "Recursive": True}
    try:
        r = requests.get(
            url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_songs failed: {e}", exc_info=True)
        return []


def get_playlist_by_name(playlist_name):
    """Finds a Jellyfin playlist by its exact name using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True, "Name": playlist_name}
    try:
        r = requests.get(
            url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        playlists = r.json().get("Items", [])
        return playlists[0] if playlists else None
    except Exception as e:
        logger.error(
            f"Jellyfin get_playlist_by_name failed for '{playlist_name}': {e}",
            exc_info=True,
        )
        return None


def create_playlist(base_name, item_ids):
    """Creates a new playlist on Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Playlists"
    body = {"Name": base_name, "Ids": item_ids, "UserId": config.JELLYFIN_USER_ID}
    try:
        r = requests.post(
            url, headers=config.HEADERS, json=body, timeout=REQUESTS_TIMEOUT
        )
        if r.ok:
            logger.info("✅ Created Jellyfin playlist '%s'", base_name)
    except Exception as e:
        logger.error(
            "Exception creating Jellyfin playlist '%s': %s", base_name, e, exc_info=True
        )


def get_all_playlists():
    """Fetches all playlists from Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Users/{config.JELLYFIN_USER_ID}/Items"
    params = {"IncludeItemTypes": "Playlist", "Recursive": True}
    try:
        r = requests.get(
            url, headers=config.HEADERS, params=params, timeout=REQUESTS_TIMEOUT
        )
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(f"Jellyfin get_all_playlists failed: {e}", exc_info=True)
        return []


def delete_playlist(playlist_id):
    """Deletes a playlist on Jellyfin using admin credentials."""
    url = f"{config.JELLYFIN_URL}/Items/{playlist_id}"
    try:
        r = requests.delete(url, headers=config.HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(
            f"Exception deleting Jellyfin playlist ID {playlist_id}: {e}", exc_info=True
        )
        return False


# --- USER-SPECIFIC JELLYFIN FUNCTIONS ---
def get_top_played_songs(limit, user_creds=None):
    """Fetches the top N most played songs from Jellyfin for a specific user."""
    user_id = user_creds.get("user_id") if user_creds else config.JELLYFIN_USER_ID
    token = user_creds.get("token") if user_creds else config.JELLYFIN_TOKEN
    if not user_id or not token:
        raise ValueError("Jellyfin User ID and Token are required.")

    url = f"{config.JELLYFIN_URL}/Users/{user_id}/Items"
    headers = {"X-Emby-Token": token}
    params = {
        "IncludeItemTypes": "Audio",
        "SortBy": "PlayCount",
        "SortOrder": "Descending",
        "Recursive": True,
        "Limit": limit,
        "Fields": "UserData,Path",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("Items", [])
    except Exception as e:
        logger.error(
            f"Jellyfin get_top_played_songs failed for user {user_id}: {e}",
            exc_info=True,
        )
        return []


def get_last_played_time(item_id, user_creds=None):
    """Fetches the last played time for a specific track from Jellyfin for a specific user."""
    user_id = user_creds.get("user_id") if user_creds else config.JELLYFIN_USER_ID
    token = user_creds.get("token") if user_creds else config.JELLYFIN_TOKEN
    if not user_id or not token:
        raise ValueError("Jellyfin User ID and Token are required.")

    url = f"{config.JELLYFIN_URL}/Users/{user_id}/Items/{item_id}"
    headers = {"X-Emby-Token": token}
    params = {"Fields": "UserData"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json().get("UserData", {}).get("LastPlayedDate")
    except Exception as e:
        logger.error(
            f"Jellyfin get_last_played_time failed for item {item_id}, user {user_id}: {e}",
            exc_info=True,
        )
        return None


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    """Creates a new instant playlist on Jellyfin for a specific user."""
    token = user_creds.get("token") if user_creds else config.JELLYFIN_TOKEN
    if not token:
        raise ValueError("Jellyfin Token is required.")

    identifier = (
        user_creds.get("user_identifier") if user_creds else config.JELLYFIN_USER_ID
    )
    if not identifier:
        raise ValueError("Jellyfin User Identifier is required.")

    user_id = resolve_user(identifier, token)

    final_playlist_name = f"{playlist_name.strip()}_instant"
    url = f"{config.JELLYFIN_URL}/Playlists"
    headers = {"X-Emby-Token": token}
    body = {"Name": final_playlist_name, "Ids": item_ids, "UserId": user_id}
    try:
        r = requests.post(url, headers=headers, json=body, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(
            "Exception creating Jellyfin instant playlist '%s' for user %s: %s",
            playlist_name,
            user_id,
            e,
            exc_info=True,
        )
        return None
