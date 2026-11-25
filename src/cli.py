"""Command line options that several scripts share."""

from src.signal_analysis import SignalSettings


def add_signal_arguments(parser) -> None:
    """Add the options of the signal analysis (smoothing, trend, gaps, peaks, outliers) to an argparse parser."""
    signal_group = parser.add_argument_group("signal analysis")
    signal_group.add_argument(
        "--smooth-window",
        type=float,
        default=0.2,
        help="length of the smoothing window in seconds (default 0.2). A longer window smooths more "
        "but also lowers fast fluctuations",
    )
    signal_group.add_argument(
        "--trend-order",
        type=int,
        default=1,
        help="order of the trend that is taken away to get the fluctuation: 0 is the mean, 1 a straight line "
        "(default 1). Use a higher order for a speed that changes a lot, for example a disc slowing down",
    )
    signal_group.add_argument(
        "--max-gap-frames",
        type=int,
        default=3,
        help="lost frames in a row that are filled in by interpolation for the analysis (default 3). "
        "A longer gap ends the stretch that is analysed",
    )
    signal_group.add_argument(
        "--min-peak-percent",
        type=float,
        default=0.5,
        help="smallest spectrum peak that is listed, as a percentage of the mean speed (default 0.5). Smaller "
        "fluctuations are not trustworthy from a phone video",
    )
    signal_group.add_argument(
        "--keep-outliers", action="store_true", help="do not replace speed values that are far from the smooth curve"
    )


def signal_settings_from_args(args) -> SignalSettings:
    """Build the signal analysis settings from the command line."""
    return SignalSettings(
        smooth_window_s=args.smooth_window,
        trend_order=args.trend_order,
        max_gap_frames=args.max_gap_frames,
        min_peak_percent=args.min_peak_percent,
        remove_outliers=not args.keep_outliers,
    )
