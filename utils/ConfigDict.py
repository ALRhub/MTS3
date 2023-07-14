import warnings


class ConfigDict:
    """Dictionary like class for setting, storing and accessing hyperparameters. Provides additional features:
      - setting and accessing of hyperparameters with "." (like attrdict)
      - finalize_adding(): No elements can be added after that call.
      - finalize_modifying(): Freezes the dict, no elements can be modified (or added)

      Intended use:
      1.) Implement a static method "get_default_config" for your approach, call finalize_adding() before you return
          the dict.
      2.) The user can now call the get_default_config to get the default parameters. Those can still be added but
          no new parameters can be added. This prevents (accidental) adding of parameters that are never used
      3.) Pass the config via the __init__ to your algorithm. Call finalize_modifying() immediately in the __init__ to
          ensure the hyperparameters stay fixed from now on.
    """

    def __init__(self, **kwargs):
        """
        the config dict will be initialized with all key value pairs in kwargs
        """
        self._adding_permitted = True
        self._modifying_permitted = True
        self._c_dict = {**kwargs}
        self._initialized = True

    def __setattr__(self, key, value):
        if "_initialized" in self.__dict__:
            if self._adding_permitted:
                self._c_dict[key] = value
            else:
                if self._modifying_permitted and key in self._c_dict.keys():
                    self._c_dict[key] = value
                elif key in self._c_dict.keys():
                    raise AssertionError("Tried modifying existing parameter after modifying finalized")
                else:
                    raise AssertionError("Tried to add parameter after adding finalized")
        else:
            self.__dict__[key] = value

    def __len__(self):
        return len(self._c_dict)

    def __getattr__(self, item):
        if "_initialized" in self.__dict__ and item in self._c_dict.keys():
            return self._c_dict[item]
        else:
            raise AssertionError("Tried accessing non existing parameter '" + str(item) + "'")

    def __getitem__(self, item):
        return self._c_dict[item]

    @property
    def raw_dict(self):
        return self._c_dict

    @property
    def adding_permitted(self):
        return self.__dict__["_adding_permitted"]

    @property
    def modifying_permitted(self):
        return self.__dict__["_modifying_permitted"]

    def finalize_adding(self):
        self.__dict__["_adding_permitted"] = False

    def finalize_modifying(self):
        if self.__dict__["_adding_permitted"]:
            warnings.warn("ConfigDict.finalize_modifying called while adding still allowed - also deactivating adding!")
            self.__dict__["_adding_permitted"] = False
        self.__dict__["_modifying_permitted"] = False

    def keys(self):
        return self._c_dict.keys()

    def items(self):
        return self._c_dict.items()
