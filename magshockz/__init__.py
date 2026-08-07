"""MagShockZ — analysis for the Magnetized Collisionless Shocks on Z experiment.

Three simulation codes feed this package, and the layout follows the stage first
and the code second::

    magshockz.common            shared by two or more codes
    magshockz.init.warpx        render and verify a WarpX deck
    magshockz.analysis.flash    FLASH post-processing
    magshockz.analysis.osiris   OSIRIS post-processing
    magshockz.analysis.warpx    WarpX post-processing

Subpackages are imported explicitly rather than re-exported here, so that reading
an import tells you which stage and which code a name belongs to, and so that
importing one module does not drag in the OSIRIS or yt stacks.
"""

__version__ = "0.1.0"
