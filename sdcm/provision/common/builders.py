# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright (c) 2021 ScyllaDB

from typing import Union

from pydantic import BaseModel, ConfigDict  # pylint: disable=no-name-in-module


OptionalType = type(Union[str, None])


class AttrBuilder(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_properties(cls):
        return [prop for prop in dir(cls) if isinstance(getattr(cls, prop), property) and prop[0] != '_']

    @property
    def _exclude_by_default(self):
        exclude_fields = []
        for field_name, field in self.__fields__.items():
            if not field.field_info.extra.get('as_dict', True):
                exclude_fields.append(field_name)
        return set(exclude_fields)

    def dict(
        self,
        *,
        include: Union['MappingIntStrAny', 'AbstractSetIntStr'] = None,  # noqa: F821
        exclude: Union['MappingIntStrAny', 'AbstractSetIntStr'] = None,  # noqa: F821
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> 'DictStrAny':  # noqa: F821
        """
        Pydantic does not treat properties as fields, so you can't get their values when call dict
        This function is to enable property extraction
        """
        if exclude is None:
            exclude = self._exclude_by_default
        attribs = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none
        )
        props = self.get_properties()
        if include:
            props = [prop for prop in props if prop in include]
        if exclude:
            props = [prop for prop in props if prop not in exclude]
        if props:
            if exclude_unset or exclude_none or exclude_defaults:
                for prop in props:
                    prop_value = getattr(self, prop)
                    if prop_value is not None:
                        attribs[prop] = prop_value
            else:
                attribs.update({prop: getattr(self, prop) for prop in props})
        return attribs
