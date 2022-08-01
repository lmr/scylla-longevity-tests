from typing import NamedTuple


class GCERegions(NamedTuple):
    us_east1: str = "us-east1"
    us_west1: str = "us-west1"


GCE_REGIONS = GCERegions()
