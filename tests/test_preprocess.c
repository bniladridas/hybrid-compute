#include <stdio.h>
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#else
#include <dirent.h>
#include <unistd.h> // for access
#endif

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <assert.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int count_files_in_dir(const char* dir_path) {
#ifdef _WIN32
    char search_path[1024];
    snprintf(search_path, sizeof(search_path), "%s\\*", dir_path);
    WIN32_FIND_DATA find_data;
    HANDLE hFind = FindFirstFile(search_path, &find_data);
    if (hFind == INVALID_HANDLE_VALUE) return -1;
    int count = 0;
    do {
        if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) count++;
    } while (FindNextFile(hFind, &find_data));
    FindClose(hFind);
    return count;
#else
    DIR* dir = opendir(dir_path);
    if (!dir) return -1;
    int count = 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) count++;
    }
    closedir(dir);
    return count;
#endif
}

int main() {
    // Create test image directory and image
    int ret;
    ret = system("mkdir -p test_images");
    (void)ret;
    ret = system("python3 ../create_test_image.py");
    (void)ret;

    // Run preprocess_c (in same directory)
    ret = system("./preprocess_c test_images test_output_tiles");
    (void)ret;

    // Check if output directory exists
    struct stat st;
    assert(stat("test_output_tiles", &st) == 0 && S_ISDIR(st.st_mode));
    (void)st;

    // Check number of tiles: 256x256 / 64x64 = 4x4 = 16
    int file_count = count_files_in_dir("test_output_tiles");
    assert(file_count == 16);
    (void)file_count;

    // Load one tile and check dimensions
    int width, height, channels;
    unsigned char* tile = stbi_load("test_output_tiles/test_tile_0.jpg", &width, &height, &channels, 0);
    assert(tile != NULL);
    assert(width == 64);
    assert(height == 64);
    assert(channels == 3);
    stbi_image_free(tile);

    // Clean up
    ret = system("rm -rf test_images test_output_tiles");
    (void)ret;

    printf("All preprocess_c tests passed!\n");
    return 0;
}
