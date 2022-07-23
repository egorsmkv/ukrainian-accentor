from torch.package import PackageImporter
from os.path import dirname


# import
_importer = PackageImporter(f"{dirname(__file__)}/accentor-lite.pt")
_accentor = _importer.load_pickle("uk-accentor", "model")


# module methods
process = _accentor.process

del PackageImporter, dirname, _importer

__all__ = ["process"]