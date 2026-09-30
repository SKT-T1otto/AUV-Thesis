"""D2's original search controller with snapshot-safe attribute delegation."""
from chapter3_bser.experiments.safe_search_v1.runtime import SearchController


class D2SearchController(SearchController):
    def __getattr__(self, name):
        # The snapshot unpickler allocates empty objects before filling fields.
        # Reading self.inner there would recursively re-enter __getattr__.
        return getattr(object.__getattribute__(self, "inner"), name)
