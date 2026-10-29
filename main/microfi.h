
#ifndef MICROFI
#define MICROFI

#include "stdint.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "esp_timer.h"

// Golden
void save_load_golden(int fi_method, tflite::MicroInterpreter* interpreter);
extern uint8_t golden_saved;
extern uint32_t current_target_index;
extern esp_timer_handle_t oneshot_timer;

// Tools
void generate_fault_list();

// RegisterFI
void create_timer();

static void oneshot_timer_callback(void* arg);

void read_stack(void *arg);

// Structure to hold task arguments
struct InferenceTaskArgs {
    tflite::MicroInterpreter* interpreter;
    TaskHandle_t mainTaskHandle;
};

// MemoryFI RAM
TfLiteStatus MemoryFI_RAM(int i, tflite::MicroInterpreter* interpreter);

TfLiteStatus MemoryFI_Flash(int i, tflite::MicroInterpreter* interpreter);

void change_random_bit_parameter();

void restore_changed_bit_parameter();

// Results
void analyze_results(int i, int8_t output_data[], int save_interval = 1, int print_interval = 100, int verbose = 2);

void save_results();

void print_results(int i, int predicted_label, int8_t max_score, int verbose = 1);

typedef struct {
    float mfr;
    float std_err;
} Statistics;

Statistics calculation(int j);

#endif // MICROFI