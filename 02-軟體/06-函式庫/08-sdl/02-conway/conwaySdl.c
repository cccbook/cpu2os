/*
 * Conway's Game of Life using SDL2 in C99
 *
 * The grid size is dynamically scaled, and the cells are displayed in two
 * states: dead and alive. The application uses a double-buffered grid to
 * update the cell states based on the classic rules of Conway's Game of Life:
 * - Any live cell with fewer than two live neighbors dies (underpopulation).
 * - Any live cell with two or three live neighbors lives on to the next
 *   generation.
 * - Any live cell with more than three live neighbors dies (overpopulation).
 * - Any dead cell with exactly three live neighbors becomes a live cell
 *   (reproduction).
 *
 * Requirements
 * - SDL2 library ('libsdl2-dev' package on Debian-based systems).
 * - C compiler (e.g., gcc).
 *
 * Compilation
 *   gcc -o main main.c $(sdl2-config --cflags --libs)
 *
 * You can also run in fullscreen mode by setting the environment variable:
 *   SDL_FULLSCREEN=1 ./main
 */

#include <SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FPS 30
#define SLEEPTIME (1000 / FPS)

typedef uint32_t pixel_t;

typedef enum { DEAD, ALIVE, N_STATES } state_t;

typedef struct {
    state_t state;
} cell_t;

static pixel_t colors[N_STATES] = {[DEAD] = 0xffffffff, [ALIVE] = 0x404040ff};

static cell_t *cell_grid_a = NULL;
static cell_t *cell_grid_b = NULL;
static pixel_t *pixels = NULL;

enum { WIDTH = 800, HEIGHT = 600 };
static int window_w = WIDTH, window_h = HEIGHT;
static int texture_w = WIDTH / 4, texture_h = HEIGHT / 4;

static SDL_Window *window = NULL;
static SDL_Renderer *renderer = NULL;
static SDL_Texture *texture = NULL;

#define grid_a(x, y) cell_grid_a[(texture_w * (y)) + (x)]
#define grid_b(x, y) cell_grid_b[(texture_w * (y)) + (x)]
#define pixel(x, y) pixels[(texture_w * (y)) + (x)]

#define set_cell_state(x, y, s)  \
    do {                         \
        grid_a(x, y).state = s;  \
        pixel(x, y) = colors[s]; \
    } while (0)

static void swap(cell_t **a, cell_t **b)
{
    cell_t *tmp = *a;
    *a = *b, *b = tmp;
}

static void seed_cell_grid(void)
{
    int dx = texture_w / 4, dy = texture_h / 4;
    int x_min = dx, x_max = texture_w - dx;
    int y_min = dy, y_max = texture_h - dy;

    for (int y = y_min; y < y_max; y++)
        for (int x = x_min; x < x_max; x++)
            set_cell_state(x, y, ALIVE);

    SDL_UpdateTexture(texture, NULL, pixels, texture_w * sizeof(pixel_t));
}

static void transition(cell_t *cell, int count)
{
    if (cell->state == ALIVE && (count < 2 || count > 3))
        cell->state = DEAD;
    else if (cell->state == DEAD && count == 3)
        cell->state = ALIVE;
}

static cell_t next_cell(int x, int y)
{
    int count = 0;
    cell_t next = grid_a(x, y);

    if (x > 0 && y > 0 && x < texture_w - 1 && y < texture_h - 1) {
        count += grid_a(x - 1, y - 1).state;
        count += grid_a(x, y - 1).state;
        count += grid_a(x + 1, y - 1).state;
        count += grid_a(x + 1, y).state;
        count += grid_a(x + 1, y + 1).state;
        count += grid_a(x, y + 1).state;
        count += grid_a(x - 1, y).state;
        count += grid_a(x - 1, y + 1).state;
    }

    transition(&next, count);
    return next;
}

static void eval_cell_grid(void)
{
    for (int y = 0; y < texture_h; y++) {
        for (int x = 0; x < texture_w; x++) {
            cell_t next = next_cell(x, y);
            grid_b(x, y) = next;
            pixel(x, y) = colors[next.state];
        }
    }

    SDL_UpdateTexture(texture, NULL, pixels, texture_w * sizeof(pixel_t));
    swap(&cell_grid_a, &cell_grid_b);
}

int main(void)
{
    srand((unsigned int) time(NULL));
    int ret = EXIT_SUCCESS;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    if (SDL_CreateWindowAndRenderer(0, 0, SDL_WINDOW_HIDDEN, &window,
                                    &renderer) != 0) {
        fprintf(stderr, "SDL_CreateWindowAndRenderer failed: %s\n",
                SDL_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    pixels = malloc(texture_w * texture_h * sizeof(pixel_t));
    cell_grid_a = calloc(texture_w * texture_h, sizeof(cell_t));
    cell_grid_b = calloc(texture_w * texture_h, sizeof(cell_t));

    if (!pixels || !cell_grid_a || !cell_grid_b) {
        fprintf(stderr, "Memory allocation failed\n");
        ret = EXIT_FAILURE;
    }

    memset(pixels, colors[DEAD], texture_w * texture_h * sizeof(pixel_t));

    texture =
        SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
                          SDL_TEXTUREACCESS_STREAMING, texture_w, texture_h);
    if (!texture) {
        fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());
        ret = EXIT_FAILURE;
        goto cleanup;
    }

    SDL_SetTextureBlendMode(texture, SDL_BLENDMODE_BLEND);
    SDL_Rect texture_rect = {.x = 0, .y = 0, .w = window_w, .h = window_h};

    SDL_SetWindowTitle(window, "Conway's Game of Life");
    SDL_SetWindowSize(window, window_w, window_h);
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED,
                          SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window);

    seed_cell_grid();

    SDL_bool done = SDL_FALSE;
    while (!done) {
        SDL_RenderCopy(renderer, texture, NULL, &texture_rect);
        SDL_RenderPresent(renderer);
        SDL_Delay(SLEEPTIME);
        eval_cell_grid();

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT ||
                (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_q))
                done = SDL_TRUE;
        }
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

cleanup:
    free(pixels);
    free(cell_grid_a);
    free(cell_grid_b);
    SDL_Quit();

    return ret;
}