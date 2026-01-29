"""
SessionManager: Single-user session management for Active Learning Dashboard.

This module provides session management to detect and prevent multiple
browser tabs from running the dashboard simultaneously. It uses a simple
file-based heartbeat mechanism optimized for single-user operation.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages single-user session detection and heartbeat.
    
    The SessionManager prevents multiple browser tabs from running the
    dashboard simultaneously by using a file-based session lock with
    heartbeat mechanism. This is optimized for single-user operation.
    
    Key features:
    - File-based session detection
    - Heartbeat mechanism to detect stale sessions
    - Graceful session acquisition and release
    - Integration with Streamlit lifecycle
    """
    
    SESSION_TIMEOUT_SECONDS = 30
    HEARTBEAT_INTERVAL_SECONDS = 10
    
    def __init__(self):
        """
        Initialize SessionManager with unique session ID.
        
        Creates a unique session identifier and sets up the session file path
        in the user's home directory.
        """
        self._session_id = str(uuid.uuid4())
        self._session_file = Path.home() / ".al_dashboard_session"
        self._is_active = False
        
        logger.info(f"SessionManager initialized with session ID: {self._session_id[:8]}...")
    
    def acquire_session(self) -> bool:
        """
        Attempt to acquire exclusive session.
        
        Checks if another session is active and acquires the session if
        no other active session exists. Uses heartbeat mechanism to
        detect stale sessions.
        
        Returns:
            True if session acquired successfully, False if another session is active
        """
        try:
            # Check if session file exists
            if self._session_file.exists():
                existing_session = self._read_session_file()
                
                if existing_session and self._is_session_active(existing_session):
                    # Another active session exists
                    logger.warning(f"Active session detected: {existing_session.get('session_id', 'unknown')[:8]}...")
                    return False
                else:
                    # Stale session, can be overridden
                    logger.info("Stale session detected, acquiring session")
            
            # Acquire session by writing our session data
            self._write_session_file()
            self._is_active = True
            
            logger.info(f"Session acquired: {self._session_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error acquiring session: {e}")
            # Fail open for robustness - allow session if file operations fail
            self._is_active = True
            return True
    
    def update_heartbeat(self) -> bool:
        """
        Update session heartbeat timestamp.
        
        Should be called periodically to indicate the session is still active.
        This prevents other instances from taking over the session.
        
        Returns:
            True if heartbeat updated successfully, False otherwise
        """
        if not self._is_active:
            logger.warning("Attempted to update heartbeat for inactive session")
            return False
        
        try:
            self._write_session_file()
            logger.debug(f"Heartbeat updated for session: {self._session_id[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error updating heartbeat: {e}")
            return False
    
    def release_session(self) -> None:
        """
        Release the session and cleanup session file.
        
        Should be called when the application is shutting down or
        when the session is no longer needed.
        """
        if not self._is_active:
            return
        
        try:
            if self._session_file.exists():
                # Verify this is our session before deleting
                existing_session = self._read_session_file()
                if existing_session and existing_session.get('session_id') == self._session_id:
                    self._session_file.unlink()
                    logger.info(f"Session released: {self._session_id[:8]}...")
                else:
                    logger.warning("Session file doesn't match our session ID, not deleting")
            
            self._is_active = False
            
        except Exception as e:
            logger.error(f"Error releasing session: {e}")
    
    def is_session_active(self) -> bool:
        """
        Check if this session is currently active.
        
        Returns:
            True if session is active, False otherwise
        """
        return self._is_active
    
    def get_session_info(self) -> dict:
        """
        Get information about the current session.
        
        Returns:
            Dictionary with session information
        """
        return {
            'session_id': self._session_id,
            'is_active': self._is_active,
            'session_file': str(self._session_file),
            'timeout_seconds': self.SESSION_TIMEOUT_SECONDS
        }
    
    def check_other_sessions(self) -> Optional[dict]:
        """
        Check if other active sessions exist.
        
        Returns:
            Dictionary with other session info if found, None otherwise
        """
        try:
            if not self._session_file.exists():
                return None
            
            existing_session = self._read_session_file()
            if not existing_session:
                return None
            
            # If it's our session, no other session
            if existing_session.get('session_id') == self._session_id:
                return None
            
            # Check if the other session is active
            if self._is_session_active(existing_session):
                return {
                    'session_id': existing_session.get('session_id', 'unknown'),
                    'last_heartbeat': existing_session.get('heartbeat'),
                    'is_stale': False
                }
            else:
                return {
                    'session_id': existing_session.get('session_id', 'unknown'),
                    'last_heartbeat': existing_session.get('heartbeat'),
                    'is_stale': True
                }
                
        except Exception as e:
            logger.error(f"Error checking other sessions: {e}")
            return None
    
    # Private helper methods
    
    def _read_session_file(self) -> Optional[dict]:
        """
        Read session data from file.
        
        Returns:
            Dictionary with session data or None if file doesn't exist or is invalid
        """
        try:
            if not self._session_file.exists():
                return None
            
            session_data = json.loads(self._session_file.read_text())
            return session_data
            
        except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
            logger.warning(f"Error reading session file: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading session file: {e}")
            return None
    
    def _write_session_file(self) -> None:
        """
        Write current session data to file.
        
        Raises:
            Exception: If file write fails
        """
        session_data = {
            'session_id': self._session_id,
            'heartbeat': datetime.now().isoformat(),
            'pid': None,  # Could add process ID if needed
            'version': '1.0'
        }
        
        # Ensure parent directory exists
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write session data atomically
        temp_file = self._session_file.with_suffix('.tmp')
        temp_file.write_text(json.dumps(session_data, indent=2))
        temp_file.replace(self._session_file)
    
    def _is_session_active(self, session_data: dict) -> bool:
        """
        Check if a session is still active based on heartbeat.
        
        Args:
            session_data: Dictionary with session information
            
        Returns:
            True if session is considered active, False if stale
        """
        try:
            heartbeat_str = session_data.get('heartbeat')
            if not heartbeat_str:
                return False
            
            # Parse heartbeat timestamp
            heartbeat = datetime.fromisoformat(heartbeat_str)
            now = datetime.now()
            
            # Check if heartbeat is within timeout window
            time_diff = now - heartbeat
            is_active = time_diff.total_seconds() < self.SESSION_TIMEOUT_SECONDS
            
            logger.debug(f"Session heartbeat check: {time_diff.total_seconds():.1f}s ago, active: {is_active}")
            return is_active
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing session heartbeat: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry."""
        if self.acquire_session():
            return self
        else:
            raise RuntimeError("Failed to acquire session - another instance may be running")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release_session()
    
    def __del__(self):
        """Destructor to ensure session cleanup."""
        try:
            if self._is_active:
                self.release_session()
        except Exception:
            pass  # Ignore errors in destructor


# Utility functions for Streamlit integration

def create_session_manager() -> SessionManager:
    """
    Create a new SessionManager instance.
    
    Returns:
        SessionManager instance
    """
    return SessionManager()


def check_session_conflict() -> Optional[dict]:
    """
    Check if there are conflicting sessions without acquiring one.
    
    Returns:
        Dictionary with conflict info if found, None otherwise
    """
    temp_manager = SessionManager()
    return temp_manager.check_other_sessions()


def cleanup_stale_sessions() -> bool:
    """
    Clean up stale session files.
    
    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        session_file = Path.home() / ".al_dashboard_session"
        
        if not session_file.exists():
            return True
        
        # Read existing session
        try:
            session_data = json.loads(session_file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            # Invalid file, remove it
            session_file.unlink()
            return True
        
        # Check if session is stale
        temp_manager = SessionManager()
        if not temp_manager._is_session_active(session_data):
            session_file.unlink()
            logger.info("Cleaned up stale session file")
        
        return True
        
    except Exception as e:
        logger.error(f"Error cleaning up stale sessions: {e}")
        return False


# Integration helpers for Streamlit

def get_session_warning_message(other_session: dict) -> str:
    """
    Get user-friendly warning message for session conflicts.
    
    Args:
        other_session: Dictionary with other session information
        
    Returns:
        Formatted warning message
    """
    session_id = other_session.get('session_id', 'unknown')[:8]
    last_heartbeat = other_session.get('last_heartbeat', 'unknown')
    is_stale = other_session.get('is_stale', False)
    
    if is_stale:
        return (
            f"⚠️ **Stale session detected** (ID: {session_id})\n\n"
            f"Last activity: {last_heartbeat}\n\n"
            "This session appears to be inactive. You can safely proceed, "
            "but please close any other browser tabs running the dashboard."
        )
    else:
        return (
            f"🚫 **Another dashboard session is active** (ID: {session_id})\n\n"
            f"Last activity: {last_heartbeat}\n\n"
            "Please close the other browser tab and refresh this page to continue. "
            "Running multiple instances simultaneously can cause conflicts."
        )


def get_session_instructions() -> str:
    """
    Get instructions for resolving session conflicts.
    
    Returns:
        Formatted instruction text
    """
    return """
    **To resolve this issue:**
    
    1. **Close other tabs** - Look for other browser tabs running the Active Learning Dashboard
    2. **Wait a moment** - Allow up to 30 seconds for the other session to timeout
    3. **Refresh this page** - Click the refresh button or press F5
    4. **Contact support** - If the issue persists, there may be a technical problem
    
    **Why does this happen?**
    
    The dashboard is designed for single-user operation to prevent conflicts between
    multiple training sessions. Only one browser tab can control the active learning
    process at a time.
    """