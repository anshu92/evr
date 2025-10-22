"""Setup registry for managing trading setups."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from .base import BaseSetup


class SetupRegistry:
    """Registry for managing trading setups."""
    
    def __init__(self):
        """Initialize setup registry."""
        self._setups: Dict[str, Type[BaseSetup]] = {}
        self._instances: Dict[str, BaseSetup] = {}
    
    def register(self, name: str, setup_class: Type[BaseSetup]) -> None:
        """Register a setup class.
        
        Args:
            name: Setup name
            setup_class: Setup class
        """
        self._setups[name] = setup_class
    
    def unregister(self, name: str) -> None:
        """Unregister a setup.
        
        Args:
            name: Setup name
        """
        self._setups.pop(name, None)
        self._instances.pop(name, None)
    
    def get_setup(self, name: str, **kwargs) -> BaseSetup:
        """Get a setup instance.
        
        Args:
            name: Setup name
            **kwargs: Setup parameters
            
        Returns:
            Setup instance
            
        Raises:
            KeyError: If setup is not registered
        """
        if name not in self._setups:
            raise KeyError(f"Setup '{name}' not registered")
        
        # Create instance if not exists or parameters changed
        if name not in self._instances or self._instances[name].parameters != kwargs:
            self._instances[name] = self._setups[name](name, **kwargs)
        
        return self._instances[name]
    
    def list_setups(self) -> List[str]:
        """List all registered setups.
        
        Returns:
            List of setup names
        """
        return list(self._setups.keys())
    
    def has_setup(self, name: str) -> bool:
        """Check if setup is registered.
        
        Args:
            name: Setup name
            
        Returns:
            True if setup is registered
        """
        return name in self._setups
    
    def get_setup_info(self, name: str) -> Optional[dict]:
        """Get setup information.
        
        Args:
            name: Setup name
            
        Returns:
            Setup information or None if not found
        """
        if name not in self._setups:
            return None
        
        setup_class = self._setups[name]
        return {
            'name': name,
            'class': setup_class.__name__,
            'module': setup_class.__module__,
            'doc': setup_class.__doc__,
        }
    
    def clear(self) -> None:
        """Clear all registered setups."""
        self._setups.clear()
        self._instances.clear()
    
    def get_all_setups(self) -> Dict[str, BaseSetup]:
        """Get all setup instances.
        
        Returns:
            Dictionary mapping setup names to instances
        """
        return self._instances.copy()


# Global registry instance
registry = SetupRegistry()
