/*
 * This program demonstrates a simple Object-Oriented Programming (OOP)
 * approach in C using function pointers and structures. It renders pixel
 * effects onto an RGBA32 buffer using SDL2 for window management and display.
 *
 * Supported effects:
 * - Bitwise operations pattern
 * - Mandelbrot fractal visualization
 *
 * Key controls:
 * - Press 'Tab' to switch between different pixel effects.
 * - Press 'Q' or close the window to quit the program.
 */

#include <SDL.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint8_t r, g, b, a;
} color_t;

/* Effect structure */
typedef struct effect {
    void (*render)(struct effect *self, color_t *canvas, int width, int height);
    color_t (*draw_pixel)(int x, int y, int width, int height);
} effect_t;

/* Generic render function for effects */
static void effect_render(effect_t *self,
                          color_t *canvas,
                          int width,
                          int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            canvas[x + y * width] = self->draw_pixel(x, y, width, height);
        }
    }
}

/* Bitwise operations */
static color_t bitop_pixel(int x, int y, int width, int height)
{
    (void) width;
    (void) height;
    return (color_t) {x ^ y, x & y, x | y, 255};
}

effect_t effect_bitop = {
    .render = effect_render,
    .draw_pixel = bitop_pixel,
};

/* Mandelbrot fractal */
static color_t mandelbrot_pixel(int x, int y, int width, int height)
{
    float cx = x / (float) width * 2.0f - 1.0f;
    float cy = y / (float) height * 2.0f - 1.0f;

    float px = 0, py = 0;
    int i;
    for (i = 0; i < 16; i++) {
        float new_px = px * px - py * py + cx;
        float new_py = 2.0f * px * py + cy;
        px = new_px, py = new_py;
        if ((px * px + py * py) > 4.0f)
            break;
    }
    uint8_t intensity = (i * 16) % 255;
    return (color_t) {intensity, intensity, intensity, 255};
}

effect_t effect_mandelbrot = {
    .render = effect_render,
    .draw_pixel = mandelbrot_pixel,
};

/* Available effects */
effect_t *effects[] = {
    &effect_mandelbrot,
    &effect_bitop,
};

#define EFFECT_COUNT (sizeof(effects) / sizeof(effects[0]))

static void panic(void)
{
    perror(SDL_GetError());
    exit(EXIT_FAILURE);
}

/* Ensure canvas size matches the window size */
static int enforce_canvas_size(SDL_Surface **canvas, SDL_Surface *win)
{
    if (!*canvas || (*canvas)->w != win->w || (*canvas)->h != win->h) {
        SDL_Surface *new_canvas = SDL_CreateRGBSurfaceWithFormat(
            0, win->w, win->h, 32, SDL_PIXELFORMAT_RGBA32);
        if (!new_canvas)
            return -1;
        if (*canvas)
            SDL_FreeSurface(*canvas);
        *canvas = new_canvas;
    }
    return 0;
}

int main(void)
{
    SDL_Surface *canvas = NULL;

    size_t effect_index = 0;
    effect_t *current = effects[effect_index];

    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        panic();

    SDL_Window *win = SDL_CreateWindow(
        "OOP in C: Pixel Effects", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, 512, 512, SDL_WINDOW_RESIZABLE);
    if (!win)
        panic();

    SDL_Event event;
    bool running = true;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                running = false;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_q)
                    running = false;
                if (event.key.keysym.scancode == SDL_SCANCODE_TAB) {
                    effect_index = (effect_index + 1) % EFFECT_COUNT;
                    current = effects[effect_index];
                }
            }
        }

        SDL_Surface *surface = SDL_GetWindowSurface(win);
        if (!surface)
            panic();

        if (enforce_canvas_size(&canvas, surface) != 0)
            panic();

        color_t *pixels = (color_t *) canvas->pixels;

        if (SDL_MUSTLOCK(canvas))
            SDL_LockSurface(canvas);

        if (current->render)
            current->render(current, pixels, canvas->w, canvas->h);

        if (SDL_MUSTLOCK(canvas))
            SDL_UnlockSurface(canvas);

        SDL_BlitSurface(canvas, NULL, surface, NULL);
        SDL_UpdateWindowSurface(win);
    }

    if (canvas)
        SDL_FreeSurface(canvas);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
