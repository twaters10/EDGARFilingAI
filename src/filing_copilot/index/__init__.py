"""The search index: the manifest it is built from, its mapping, and the build."""

from .build import BuildReport, IndexBuildError, build_index, index_name, to_document
from .manifest import ManifestRow, manifest_path, read_manifest, write_manifest
from .mapping import index_body
from .opensearch import OpenSearchClient, OpenSearchError

__all__ = [
    "BuildReport",
    "IndexBuildError",
    "ManifestRow",
    "OpenSearchClient",
    "OpenSearchError",
    "build_index",
    "index_body",
    "index_name",
    "manifest_path",
    "read_manifest",
    "to_document",
    "write_manifest",
]
