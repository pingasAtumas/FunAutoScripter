"""FunAutoScripter — entry point and public API re-exports. Engine lives in src/."""
from src import (
    log, probe, _probe, crop_filter, parse_zoom, CACHE_VERSION, cache_key,
    decode_cmd, Cancelled, NoFlow, extract_flow, BACKEND, FIXED_THREADS,
    opencl_state, accel_report, pick_jobs, cache_path, cache_valid, load_or_extract,
    read_cache_vr_zoom,
    split_windows, periodicity_score, roi_size, choose_rois, AXIS_REFS, axis_vec,
    chapter_axis,
    velocity, to_position, normalise, enhance, post_enhance, keyframes, compute,
    write_funscript, plot, AXIS_SUFFIX, axis2_path,
    PALETTE, FONT, put, box, bar, VideoOut, draw_plot, DEBUG_VIDEOS, DEBUG_ALIASES,
    parse_debug_videos, render_debug,
    chapters_file, load_chapters, chapter_args, limit_speed, load_full,
    _full_args, flow_available, flow_for, process_chapters, detect_chapters,
    write_chapters_json, resolve_chapters,
    VIDEO_EXTS, find_videos, process, process_many, run_folder,
    build_parser, main,
)

if __name__ == "__main__":
    main()
