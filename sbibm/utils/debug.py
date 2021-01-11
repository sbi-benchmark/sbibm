import sys


def pdb_hook(type, value, tb):
    """PDB Hook

    Usage:
        import sys
        from sbibm.utils.debug import pdb_hook
        sys.excepthook = pdb_hook
    """
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        import traceback

        try:
            import ipdb as pdb

        except:
            import pdb

        traceback.print_exception(type, value, tb)
        pdb.post_mortem(tb)


def pdb_hook_always(type, value, tb):
    """PDB Hook

    Like `pdb_hook`, but always attached.

    Usage:
        import sys
        from sbibm.utils.debug import pdb_hook_always
        sys.excepthook = pdb_hook_always
    """
    import traceback

    try:
        import ipdb as pdb

    except:
        import pdb

    traceback.print_exception(type, value, tb)
    pdb.post_mortem(tb)
