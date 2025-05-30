#ifndef C63_C63_H_
#define C63_C63_H_

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#define GROUP 9

#ifndef GROUP
#error Fill in group number
#endif

#define NO_FLAGS 0
#define NO_CALLBACK NULL
#define NO_ARG NULL

/* GET_SEGMENTID(2) gives you segmentid 2 at your groups offset */
#define GET_SEGMENTID(id) ( GROUP << 16 | id )
#define SEGMENT_CLIENT GET_SEGMENTID(1)
#define SEGMENT_SERVER GET_SEGMENTID(2)

// Message sizes
#define MESSAGE_SIZE (8 * 1024 * 1024)  // 8MB buffer size

// Command definitions for signaling
enum cmd
{
    CMD_INVALID = 0,   // No command/initial state
    CMD_HELLO,         // Client sending hello
    CMD_HELLO_ACK,     // Server acknowledging hello
    CMD_DIMENSIONS,    // Client sending dimensions
    CMD_DIMENSIONS_ACK,// Server acknowledging dimensions
    CMD_QUIT,          // Signal to terminate
    CMD_DATA_READY,    // Signal that data is ready to be read
    CMD_YUV_DATA,      // Client sending YUV frame data
    CMD_YUV_DATA_ACK,  // Server acknowledging receipt of YUV data
    CMD_ENCODED_DATA,  // Server sending encoded frame data
    CMD_ENCODED_DATA_ACK// Client acknowledging receipt of encoded data
};

#define MAX_FILELENGTH 200
#define DEFAULT_OUTPUT_FILE "a.mjpg"

#define PI 3.14159265358979
#define ILOG2 1.442695040888963 // 1/log(2);

#define COLOR_COMPONENTS 3

#define Y_COMPONENT 0
#define U_COMPONENT 1
#define V_COMPONENT 2

#define YX 2
#define YY 2
#define UX 1
#define UY 1
#define VX 1
#define VY 1

/* The JPEG file format defines several parts and each part is defined by a
 marker. A file always starts with 0xFF and is then followed by a magic number,
 e.g., like 0xD8 in the SOI marker below. Some markers have a payload, and if
 so, the size of the payload is written before the payload itself. */

#define JPEG_DEF_MARKER 0xFF
#define JPEG_SOI_MARKER 0xD8
#define JPEG_DQT_MARKER 0xDB
#define JPEG_SOF_MARKER 0xC0
#define JPEG_DHT_MARKER 0xC4
#define JPEG_SOS_MARKER 0xDA
#define JPEG_EOI_MARKER 0xD9

#define HUFF_AC_ZERO 16
#define HUFF_AC_SIZE 11

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

struct yuv
{
    uint8_t *Y;
    uint8_t *U;
    uint8_t *V;
};

struct dct
{
    int16_t *Ydct;
    int16_t *Udct;
    int16_t *Vdct;
};

typedef struct yuv yuv_t;
typedef struct dct dct_t;

struct entropy_ctx
{
    FILE *fp;
    unsigned int bit_buffer;
    unsigned int bit_buffer_width;
};

struct macroblock
{
    int use_mv;
    int8_t mv_x, mv_y;
};

struct frame
{
    yuv_t *orig;        // Original input image
    yuv_t *recons;      // Reconstructed image
    yuv_t *predicted;   // Predicted frame from intra-prediction

    dct_t *residuals;   // Difference between original image and predicted frame

    struct macroblock *mbs[COLOR_COMPONENTS];
    int keyframe;
};

struct c63_common
{
    int width, height;
    int ypw, yph, upw, uph, vpw, vph;

    int padw[COLOR_COMPONENTS], padh[COLOR_COMPONENTS];

    int mb_cols, mb_rows;

    uint8_t qp;         // Quality parameter

    int me_search_range;

    uint8_t quanttbl[COLOR_COMPONENTS][64];

    struct frame *refframe;
    struct frame *curframe;

    int framenum;

    int keyframe_interval;
    int frames_since_keyframe;

    struct entropy_ctx e_ctx;
};

struct dimensions_data {
    uint32_t width;
    uint32_t height;
};

struct packet
{
    union {
        struct {
            uint32_t cmd;           // Command type
            uint32_t data_size;     // Size of data in buffer
            uint32_t y_size;        // Y size
            uint32_t u_size;        // U size
            uint32_t v_size;        // V size
        };
        uint8_t padding[64];        // Align to cache line
    } __attribute__((aligned(64)));
};

// Server segment structure
struct server_segment
{
    struct packet packet __attribute__((aligned(64)));
    char message_buffer[MESSAGE_SIZE] __attribute__((aligned(64))); // Buffer for messages
};

// Client segment structure
struct client_segment
{
    struct packet packet __attribute__((aligned(64)));
    char message_buffer[MESSAGE_SIZE] __attribute__((aligned(64))); // Buffer for messages
};

#endif  /* C63_C63_H_ */