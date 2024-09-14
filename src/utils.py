# https://github.com/Selur/VapoursynthScriptsInHybrid/blob/550cc72221848732fa36a5dfb9ad5f98a308dd6e/havsfunc.py#L5217
# https://github.com/HomeOfVapourSynthEvolution/mvsfunc/blob/865c7486ca860d323754ec4774bc4cca540a7076/mvsfunc/mvsfunc.py#L2570
import vapoursynth as vs
import math

core = vs.core


def GetPlane(clip, plane=None):
    # input clip
    if not isinstance(clip, vs.VideoNode):
        raise type_error('"clip" must be a clip!')

    # Get properties of input clip
    sFormat = clip.format
    sNumPlanes = sFormat.num_planes

    # Parameters
    if plane is None:
        plane = 0
    elif not isinstance(plane, int):
        raise type_error('"plane" must be an int!')
    elif plane < 0 or plane > sNumPlanes:
        raise value_error(f'valid range of "plane" is [0, {sNumPlanes})!')

    # Process
    return core.std.ShufflePlanes(clip, plane, vs.GRAY)


def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255


def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)


##############################
# FastLineDarken 1.4x MT MOD #
##############################
#
# Written by Vectrangle    (http://forum.doom9.org/showthread.php?t=82125)
# Didée: - Speed Boost, Updated: 11th May 2007
# Dogway - added protection option. 12-May-2011
#
# Parameters are:
#  strength (integer)   - Line darkening amount, 0-256. Default 48. Represents the _maximum_ amount
#                         that the luma will be reduced by, weaker lines will be reduced by
#                         proportionately less.
#  protection (integer) - Prevents the darkest lines from being darkened. Protection acts as a threshold.
#                         Values range from 0 (no prot) to ~50 (protect everything)
#  luma_cap (integer)   - value from 0 (black) to 255 (white), used to stop the darkening
#                         determination from being 'blinded' by bright pixels, and to stop grey
#                         lines on white backgrounds being darkened. Any pixels brighter than
#                         luma_cap are treated as only being as bright as luma_cap. Lowering
#                         luma_cap tends to reduce line darkening. 255 disables capping. Default 191.
#  threshold (integer)  - any pixels that were going to be darkened by an amount less than
#                         threshold will not be touched. setting this to 0 will disable it, setting
#                         it to 4 (default) is recommended, since often a lot of random pixels are
#                         marked for very slight darkening and a threshold of about 4 should fix
#                         them. Note if you set threshold too high, some lines will not be darkened
#  thinning (integer)   - optional line thinning amount, 0-256. Setting this to 0 will disable it,
#                         which is gives a _big_ speed increase. Note that thinning the lines will
#                         inherently darken the remaining pixels in each line a little. Default 0.
def FastLineDarkenMOD(
    c, strength=48, protection=5, luma_cap=191, threshold=4, thinning=0
):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error("FastLineDarkenMOD: this is not a clip")

    if c.format.color_family == vs.RGB:
        raise vs.Error("FastLineDarkenMOD: RGB format is not supported")

    peak = (
        (1 << c.format.bits_per_sample) - 1
        if c.format.sample_type == vs.INTEGER
        else 1.0
    )

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = GetPlane(c, 0)
    else:
        c_orig = None

    ## parameters ##
    Str = strength / 128
    lum = scale(luma_cap, peak)
    thr = scale(threshold, peak)
    thn = thinning / 16

    ## filtering ##
    exin = c.std.Maximum(threshold=peak / (protection + 1)).std.Minimum()
    thick = core.std.Expr(
        [c, exin],
        expr=[
            f"y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {Str} * x +"
        ],
    )
    if thinning <= 0:
        last = thick
    else:
        diff = core.std.Expr(
            [c, exin],
            expr=[
                f"y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {scale(127, peak)} +"
            ],
        )
        linemask = (
            diff.std.Minimum()
            .std.Expr(expr=[f"x {scale(127, peak)} - {thn} * {peak} +"])
            .std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        thin = core.std.Expr(
            [c.std.Maximum(), diff], expr=[f"x y {scale(127, peak)} - {Str} 1 + * +"]
        )
        last = core.std.MaskedMerge(thin, thick, linemask)

    if c_orig is not None:
        last = core.std.ShufflePlanes(
            [last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family
        )
    return last
