#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x264.h>

static void fail(const char *message) { fprintf(stderr, "%s\n", message); exit(2); }
static int argi(int argc, char **argv, const char *name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return atoi(argv[i + 1]);
    return fallback;
}
static const char *args(int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return fallback;
}
static void write_nals(FILE *output, x264_nal_t *nals, int count) {
    for (int i = 0; i < count; ++i)
        if (fwrite(nals[i].p_payload, 1, (size_t)nals[i].i_payload, output) != (size_t)nals[i].i_payload)
            fail("Failed to write H.264 output");
}
static int read_plane(FILE *input, uint8_t *dst, int stride, int width, int height) {
    for (int row = 0; row < height; ++row)
        if (fread(dst + row * stride, 1, (size_t)width, input) != (size_t)width) return 0;
    return 1;
}
int main(int argc, char **argv) {
    int width = argi(argc, argv, "--width", 0), height = argi(argc, argv, "--height", 0);
    int fps = argi(argc, argv, "--fps", 0), bitrate = argi(argc, argv, "--bitrate-kbps", 0);
    int keyint = argi(argc, argv, "--keyint", 30), bframes = argi(argc, argv, "--bframes", 0);
    const char *input_path = args(argc, argv, "--input", NULL);
    const char *maps_path = args(argc, argv, "--maps", NULL);
    const char *output_path = args(argc, argv, "--output", NULL);
    const char *trace_path = args(argc, argv, "--trace", NULL);
    const char *preset = args(argc, argv, "--preset", "medium");
    const char *tune = args(argc, argv, "--tune", "zerolatency");
    if (!input_path || !maps_path || !output_path || !trace_path || width < 2 || height < 2 ||
        (width & 1) || (height & 1) || fps < 1 || bitrate < 1) fail("Invalid bridge arguments");
    FILE *input = fopen(input_path, "rb"), *maps = fopen(maps_path, "rb");
    FILE *output = fopen(output_path, "wb"), *trace = fopen(trace_path, "w");
    if (!input || !maps || !output || !trace) fail("Unable to open bridge input/output file");
    x264_param_t param;
    if (x264_param_default_preset(&param, preset, tune && strcmp(tune, "none") ? tune : NULL) < 0)
        fail("x264 rejected preset/tune");
    param.i_csp = X264_CSP_I420; param.i_width = width; param.i_height = height;
    param.i_fps_num = fps; param.i_fps_den = 1; param.i_timebase_num = 1; param.i_timebase_den = fps;
    param.i_keyint_max = keyint; param.i_bframe = bframes; param.i_threads = 1;
    param.i_sync_lookahead = 0; param.rc.i_lookahead = 0;
    param.rc.i_rc_method = X264_RC_ABR; param.rc.i_bitrate = bitrate;
    param.rc.i_aq_mode = X264_AQ_VARIANCE; param.rc.f_aq_strength = 1.0f;
    param.b_annexb = 1; param.b_repeat_headers = 1;
    if (x264_param_apply_profile(&param, "high") < 0) fail("x264 rejected high profile");
    x264_t *encoder = x264_encoder_open(&param); if (!encoder) fail("x264_encoder_open failed");
    x264_picture_t picture, encoded_picture;
    if (x264_picture_alloc(&picture, X264_CSP_I420, width, height) < 0) fail("picture allocation failed");
    int mb_count = ((width + 15) / 16) * ((height + 15) / 16), frame_id = 0;
    fprintf(trace, "frame_id,pts,frame_average_crf,frame_average_crf_available\n");
    for (;;) {
        long position = ftell(input);
        if (!read_plane(input, picture.img.plane[0], picture.img.i_stride[0], width, height)) {
            fseek(input, position, SEEK_SET); int c = fgetc(input);
            if (c != EOF) fail("Truncated YUV input frame");
            break;
        }
        if (!read_plane(input, picture.img.plane[1], picture.img.i_stride[1], width / 2, height / 2) ||
            !read_plane(input, picture.img.plane[2], picture.img.i_stride[2], width / 2, height / 2))
            fail("Truncated YUV input frame");
        float *offsets = malloc((size_t)mb_count * sizeof(float)); if (!offsets) fail("offset allocation failed");
        if (fread(offsets, sizeof(float), (size_t)mb_count, maps) != (size_t)mb_count)
            fail("QP map file has fewer frames than input");
        picture.i_pts = frame_id; picture.opaque = (void *)(intptr_t)frame_id;
        picture.prop.quant_offsets = offsets; picture.prop.quant_offsets_free = free;
        x264_nal_t *nals = NULL; int nal_count = 0;
        int bytes = x264_encoder_encode(encoder, &nals, &nal_count, &picture, &encoded_picture);
        if (bytes < 0) fail("x264_encoder_encode failed");
        write_nals(output, nals, nal_count);
        if (bytes > 0) fprintf(trace, "%" PRIdPTR ",%" PRId64 ",%.8f,true\n",
                               (intptr_t)encoded_picture.opaque, encoded_picture.i_pts,
                               encoded_picture.prop.f_crf_avg);
        frame_id++;
    }
    if (fgetc(maps) != EOF) fail("QP map file has more frames than input");
    while (x264_encoder_delayed_frames(encoder)) {
        x264_nal_t *nals = NULL; int nal_count = 0;
        int bytes = x264_encoder_encode(encoder, &nals, &nal_count, NULL, &encoded_picture);
        if (bytes < 0) fail("x264 flush failed");
        write_nals(output, nals, nal_count);
        if (bytes > 0) fprintf(trace, "%" PRIdPTR ",%" PRId64 ",%.8f,true\n",
                               (intptr_t)encoded_picture.opaque, encoded_picture.i_pts,
                               encoded_picture.prop.f_crf_avg);
    }
    x264_picture_clean(&picture); x264_encoder_close(encoder);
    fclose(input); fclose(maps); fclose(output); fclose(trace);
    return 0;
}
