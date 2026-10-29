/*==============================================================================
MicroFI 0.4
Fault Injection Framework for TensorFlow Lite for Microcontrollers
Leonardo Alexandrino de Melo
Lyon Institute of Nanotechnology
==============================================================================*/

#include "microfi.h"
#include "model.h"
#include "image_input.h"
#include "main_functions.h"

#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "esp_system.h"
#include "esp_flash.h"
#include "esp_mac.h"
#include "esp_partition.h"
#include "esp_random.h"
#include "esp_task_wdt.h"
#include "spi_flash_mmap.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_intr_alloc.h"
#include <stdio.h>
#include "nvs_flash.h"
#include "nvs.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "sdkconfig.h"

#include "..\components\espressif__esp-tflite-micro\tensorflow\lite\micro\micro_interpreter_graph.h"

static const char* TAG = "example";

constexpr int number_of_targets = (sizeof(targets)/sizeof(targets[0]));
constexpr int number_of_inputs = sizeof(labels)/sizeof(labels[0]);

// Minimum value for int8 (change for other quantization)
constexpr int min_quant_value = sizeof(int);

// Stored values on flash
// Golden
static int32_t golden[number_of_inputs][number_of_outputs + 2];
static uint32_t golden_inference_ticks = sizeof(uint32_t); // Time in ms

// 0-9 - Scores, 10 - Golden Prediction, 11 - Golden Max Score

// Fault classification
static int32_t fault_type_per_target[number_of_targets][6];
// 0 - Masked, 1 - Observed Acceptable, 2 - Observed Good, 3 - Observed Warning, 4 - Observed Critical, 5 - Crashes

// Accuracy counter
static int32_t predictions[3];
// 0 - Total Predictions, 1 - Correct Faulty, 2 - Correct Golden

// Fault Injection Parameters
constexpr int fault_list_size = 1 * number_of_targets;
int injection_span = 32; // bits
int registers[] = {//5, 6, 7, 28, 29, 30, 31, // Temporary registers (5 is also frame pointer)
                  //10, 11, 12, 13, 14, 15, 16, 17 // Function arguments (10 an 11 also return values)
                  9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27 // Saved registers (8 also frame pointer)
                  };

uint32_t current_target_index = -1;
uint32_t target_byte_offset[fault_list_size];
uint8_t target_bit_offset[fault_list_size];

esp_timer_handle_t oneshot_timer;

nvs_handle_t storage_handle;

void save_load_golden(int fi_method, tflite::MicroInterpreter* interpreter){
    // Check if the flag is already set (stored values in place in flash)
    uint8_t golden_saved = 0; // Flag

    nvs_flash_init(); // Initiate Flash only once!
    nvs_open("storage", NVS_READWRITE, &storage_handle); // Open storage
    nvs_get_u8(storage_handle, "golden_saved", &golden_saved); // Get Flag
    
    if(golden_saved != 1){ // If golden is not saved, run unperturbed inference for golden
        TfLiteStatus invoke_status;
        TfLiteTensor* input = interpreter->input(0);
        TfLiteTensor* output = interpreter->output(0);
        int32_t start_ticks, end_ticks;
        
        for (int i = 0; i < number_of_inputs; ++i){
            for (int l = 0; l < input_size; l++) { // Copy data to input tensor 
                input->data.raw[l] = image_bin[l+(input_size*i)];
            }
            
            //start_ticks = esp_cpu_get_cycle_count();  
            invoke_status = interpreter->Invoke(); // Inference
            //end_ticks = esp_cpu_get_cycle_count();
            
            if (invoke_status == kTfLiteOk){
                //if((end_ticks - start_ticks) < golden_inference_ticks) golden_inference_ticks = (end_ticks - start_ticks);
                golden[i][10] = -1; // (predicted label) Store the confidence level to compare with predictions label
                golden[i][11] = min_quant_value; // (max score) Store the higher confidence result

                for (int j = 0; j < number_of_outputs; ++j) { // Read output
                    if (output->data.int8[j] > golden[i][11]) { // Search for predicted value
                        golden[i][11] = output->data.int8[j]; // Update prediction
                        golden[i][10] = j; // Update max score
                    }
                    golden[i][j] = output->data.int8[j]; // Save golden output
                    //MicroPrintf("Golden: %i =\t%i\t%.2f", j, golden[i][j],((golden[i][j]+128.0)/255.0)*100.0);
                }
            } else { MicroPrintf("Inference Failed on Golden"); esp_restart(); }
        }
        // Save to Flash
        size_t array_size = sizeof(golden);
        nvs_set_blob(storage_handle, "golden", golden, array_size); // Save golden
        golden_saved = 1;
        nvs_set_u8(storage_handle, "golden_saved", golden_saved); // Save updated flag
        //nvs_set_u32(storage_handle, "inf_ticks", golden_inference_ticks); // Save updated flag
        MicroPrintf("Golden Saved");
    }

    else{ // Flag is set, only restore preiously computed golden
        size_t array_size = sizeof(golden);
        nvs_get_blob(storage_handle, "golden", golden, &array_size);
        array_size = sizeof(predictions);
        nvs_get_blob(storage_handle, "predictions", predictions, &array_size);
        array_size = sizeof(fault_type_per_target);
        nvs_get_blob(storage_handle, "faults", fault_type_per_target, &array_size);
        //nvs_get_u32(storage_handle, "inf_time", &golden_inference_ticks); // Save updated flag

        current_target_index = (predictions[0] + 1) % number_of_targets;

        if(fi_method == 4) { // Handle crashes (or resets)
            fault_type_per_target[current_target_index][5]++;
            predictions[0]++;
            current_target_index = (predictions[0] + 1) % number_of_targets;
            nvs_set_blob(storage_handle, "faults", fault_type_per_target, array_size);
            array_size = sizeof(predictions);
            nvs_set_blob(storage_handle, "predictions", predictions, array_size);
        } else{ generate_fault_list(); }
        
        MicroPrintf("Golden and Results Restored");
    }
    nvs_commit(storage_handle); // Apply changes
    nvs_close(storage_handle); // Close storage
}

// ==================== Tools ====================

void generate_fault_list(){
    for (int i = 0; i < fault_list_size; i++) {
        current_target_index = i % number_of_targets;
        target_byte_offset[i] = parameter_tensor_positions_lookup[current_target_index][1] + // 1 = Position 2 = hex len
        (esp_random() % parameter_tensor_positions_lookup[current_target_index][2]); // Choose a random byte in the found hex string
        target_bit_offset[i] = esp_random() % 8; // Choose a random bit to flip in the chosen byte
    }
    current_target_index = 0;
    MicroPrintf("\nFault List Generated");
}

void create_timer(){
    const esp_timer_create_args_t oneshot_timer_args = {
            .callback = &oneshot_timer_callback,
            /* argument specified here will be passed to timer callback function */
            .name = "one-shot"
    };
    esp_timer_create(&oneshot_timer_args, &oneshot_timer);
}



static void oneshot_timer_callback(void* arg)
{   
    //esp_timer_handle_t periodic_timer_handle = (esp_timer_handle_t) arg;
    //ESP_ERROR_CHECK(esp_timer_stop(periodic_timer_handle)); 
        //MicroPrintf("Interrupted! ======= RegisterFI");
    //asm volatile ("mv %0, sp" : "=r"(stack_pointer)); // Save Stack Pointer
    //int32_t value = *( (int32_t *) stack_pointer + target_stack_offset ); // Read old value
    //MicroPrintf("Stack Pointer: %p", stack_pointer); 
    // Randomize fault injection location
    int target_register = registers[esp_random() % (sizeof(registers) / sizeof(registers[0]))]; // Select a number from a random index of the array
    int target_stack_offset = target_register - 36; // Stack grows downwards and upsidedown from stack_pointer[-5] for esp32c3
    int random_bit_offset = esp_random() % injection_span; // Choose a random bit to flip from 32 bits
      
    //MicroPrintf("Time Injection: %lld", esp_timer_get_time());
    *( (int32_t *) stack_pointer+target_stack_offset ) ^= (1 << random_bit_offset); // Flip the bit using XOR
    //MicroPrintf("Time Injection: %lld", esp_timer_get_time());

    //notification = 2;
    //vTaskDelay(10000);
}

void read_stack(void *arg) {
  vTaskSuspend(xTaskGetHandle("inference"));
  int32_t value;
  MicroPrintf("\nStack pointer address: %p\n", (void *)stack_pointer); // Print stack pointer address
  for (int i = -36; i <= 0; i++) {// Print all registers
    value = *( (int32_t *) stack_pointer+i ); // Read all registers
    MicroPrintf("(Index %i)(r%i) Stack value at %p: %li\n", i, i+5+31, stack_pointer+i, value); 
  }
  vTaskResume(xTaskGetHandle("inference"));
}

// ====================== MemoryFI ======================

TfLiteStatus MemoryFI_RAM(int i, tflite::MicroInterpreter* interpreter){
    //change_random_bit_parameter();
    //int32_t start_ticks, end_ticks;
    //static uint32_t elapsed_ticks = 0;

    //start_ticks = esp_cpu_get_cycle_count();
    g_model[target_byte_offset[current_target_index]] ^= (1 << target_bit_offset[current_target_index]); // Flip the bit using XOR
    //end_ticks = esp_cpu_get_cycle_count();
    //elapsed_ticks += end_ticks - start_ticks;

    TfLiteStatus invoke_status = interpreter->Invoke();

    //restore_changed_bit_parameter();
    //start_ticks = esp_cpu_get_cycle_count();
    g_model[target_byte_offset[current_target_index]] ^= (1 << target_bit_offset[current_target_index]); // Flip back the bit using XOR
    //end_ticks = esp_cpu_get_cycle_count();
    //elapsed_ticks += end_ticks - start_ticks;
    //if(!(predictions[0] % 1000)) {MicroPrintf("\nTime taken to FI: %d CPU ticks", elapsed_ticks/1000); elapsed_ticks=0;}
    
    current_target_index++;
    if(!(predictions[0] % fault_list_size)) generate_fault_list();
    return invoke_status; 
}

// Function to change a random bit in the found hex string
void change_random_bit_parameter() {
    int value = g_model[target_byte_offset[current_target_index]]; // Read old value
    MicroPrintf("\n=== Memory Fault Injection ===\nMemoryFI: Original value at %p: %i", g_model[target_byte_offset[current_target_index]], value); 

    //g_model[target_byte_offset[current_target_index]] ^= (1 << target_bit_offset[current_target_index]); // Flip the bit using XOR

    value = g_model[target_byte_offset[current_target_index]]; // Read new value
    MicroPrintf("MemoryFI: Changed bit %d in byte %zu on tensor %i, new value: %i", target_bit_offset[current_target_index], target_byte_offset[current_target_index], targets[fault_index], value);
}

void restore_changed_bit_parameter(){
    int value = g_model[target_byte_offset[current_target_index]]; // Read new value
    MicroPrintf("MemoryFI: Restoring bit %u, value on memory: %04x", target_bit_offset[current_target_index], value);
    
    //g_model[target_byte_offset[current_target_index]] ^= (1 << target_bit_offset[current_target_index]); // Flip the bit using XOR

    value = g_model[target_byte_offset[current_target_index]]; // Read new value
    MicroPrintf("MemoryFI: Restored byte %zu, value on memory: %04x", target_byte_offset[current_target_index], value); 
}

// ====================== Results ======================

void analyze_results(int i, int8_t output_data[], int save_interval, int print_interval, int verbose){
    int predicted_label = -1; // Store the confidence level to compare with predictions label
    int8_t max_score = min_quant_value; // Store the higher confidence result
    
    for (int j = 0; j < number_of_outputs; ++j) { // Read output
        if (output_data[j] > max_score) { // Search for predicted value
            max_score = output_data[j];
            predicted_label = j;
        }
    }

    if(max_score > min_quant_value){ // If a valid inference did run
        //if(fi_method == 1)fault_index = node_index_check;
        predictions[0]++; // Total predictions

        if(((max_score + 128.0) / 255.0) > 0.8 && predicted_label == labels[i])
        {predictions[1]++;} // Correct predictions by faulty DNN
        
        if(((golden[i][11] + 128.0) / 255.0) > 0.8 && golden[i][10] == labels[i])
        {predictions[2]++;} // Correct predictions by golden DNN

        if(predicted_label == golden[i][10] && max_score == golden[i][11]) fault_type_per_target[fault_index][0]++; // Masked

        else if((max_score > golden[i][11] && golden[i][10] == labels[i] && predicted_label == golden[i][10]) || 
        (predicted_label == labels[i] && golden[i][10] != labels[i])) fault_type_per_target[fault_index][1]++; // Observed Good

        else if(max_score < golden[i][11] && max_score > golden[i][11] - 13 && 
        predicted_label == golden[i][10]) fault_type_per_target[fault_index][2]++; // Observed Acceptable

        else if(max_score < golden[i][11] - 13 && predicted_label == golden[i][10]) fault_type_per_target[fault_index][3]++; // Observed Warning

        else{fault_type_per_target[fault_index][4]++;} // Observed Critical
    }

    if(!(predictions[0] % save_interval)) save_results(); // Save results on flash
    if(!(predictions[0] % print_interval) && verbose > 0) print_results(i, predicted_label, max_score, verbose);
    
    
    fault_index = predictions[0] % number_of_targets;
}

void save_results(){
    nvs_open("storage", NVS_READWRITE, &storage_handle); // Open storage
    size_t array_size = sizeof(fault_type_per_target);
    nvs_set_blob(storage_handle, "faults", fault_type_per_target, array_size);
    array_size = sizeof(predictions);
    nvs_set_blob(storage_handle, "predictions", predictions, array_size);
    nvs_commit(storage_handle); // Apply changes
    nvs_close(storage_handle); // Close storage
    MicroPrintf("\nResults Saved");
}

void print_results(int i, int predicted_label, int8_t max_score, int verbose){
    if(verbose > 2){ // Print info related to current prediction
        MicroPrintf("\nPredicted: %i\tLabel: %i\tMax score: (Golden: %i) %i", 
        predicted_label, labels[i], golden[i][11], max_score);
    }
 
    if(verbose > 1){ // Print info related to faults
        MicroPrintf("\nCorrect Classifications: %li (Golden: %li)\nAccuracy: %.2f%% (Golden: %.2f%%)\tTotal Predictions: %li",
        predictions[1], predictions[2], 100.0*(double)predictions[1]/(double)predictions[0], 
        100.0*(double)predictions[2]/(double)predictions[0], predictions[0]);
        for(int j = 0; j < number_of_targets; ++j){ // Print fault types per target
            Statistics result = calculation(j);
            MicroPrintf(" -- Target %li: Masked: %li | MFR: %.2f%% STD_ERR: %.4f%%", targets[j], fault_type_per_target[j][0], (double)result.mfr*100.0, (double)result.std_err*100.0);
            MicroPrintf("Observed: Good: %li Acceptable: %li Warning: %li Critical: %li Crashes:%li",
            fault_type_per_target[j][1], fault_type_per_target[j][2], fault_type_per_target[j][3], fault_type_per_target[j][4], fault_type_per_target[j][5]);
        }  
    } 
    
    else{ // Print only basic info to know when to stop FI
        Statistics result = calculation(0);
        MicroPrintf("Total Predictions: %li | MFR: %.4f%% STD_ERR: %.4f%%", predictions[0], (double)result.mfr*100, (double)result.std_err*100); 
    }
}

Statistics calculation(int j){ // Calculate MFR and STD_ERR to know when to stop FI
    Statistics stats;
    int total_faults = fault_type_per_target[j][0] + fault_type_per_target[j][1] + fault_type_per_target[j][2] + fault_type_per_target[j][3] + fault_type_per_target[j][4] + fault_type_per_target[j][5];
    int observed_faults = total_faults - fault_type_per_target[j][0];
    int bad_faults = observed_faults - fault_type_per_target[j][1];
    stats.mfr = (double)bad_faults / (double)total_faults; // Mismatch Failure Rate
    stats.std_err = sqrt((stats.mfr*(1-stats.mfr))/(double)total_faults); // Standard Error
    return stats;
}





