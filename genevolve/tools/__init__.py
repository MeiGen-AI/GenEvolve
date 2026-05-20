"""Search tools for GenEvolve: text search, image search.

These wrap public search-API providers. Configure via environment variables:

  - ``SERPER_API_KEY``  : api.serper.dev key for both text and image search.
  - ``SERPER_BASE_URL`` : optional, default ``https://google.serper.dev``.
  - ``IMAGE_DOWNLOAD_DIR`` : where to cache image_search downloads (default ``/tmp/genevolve_images``).

The same protocol works with any Serper-compatible HTTP gateway as long as
``/search`` returns ``organic`` text results and ``/images`` returns ``images``.
"""

from .web_search import WebTextSearchTool, ImageSearchTool

__all__ = ["WebTextSearchTool", "ImageSearchTool"]
