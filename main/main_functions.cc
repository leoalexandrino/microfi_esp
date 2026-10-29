/*==============================================================================
MicroFI 0.4
Fault Injection Framework for TensorFlow Lite for Microcontrollers
Leonardo Alexandrino de Melo
Lyon Institute of Nanotechnology
==============================================================================*/

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "microfi.h"
#include "image_input.h"
#include "model.h"

#include "string.h"
#include "esp_mac.h"
#include "esp_err.h"
#include "esp_task_wdt.h"
#include "hal/cpu_hal.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_intr_alloc.h"
#include "tensorflow/lite/micro/micro_interpreter_graph.h"
#include "esp_random.h"
#include "esp_expression_with_stack.h"

// Parameters FI
// 4 = RegisterFI, 2 = MemoryFI (RAM)
int fi_method = 4; 
// Nodes: Original: 0, 4, 8, 11, 13 TMR:  0, 4, 8, 14, 16 // tensors og=3, 5, 7, 9, 11 TMR=12,10,8,6,4
// These are the number of the nodes that refers to the main targeted layers
int targets[5] = {0, 4, 8, 11, 13}; 

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 100000; // Default for Hello World: 2000, Lenet: 16000
  uint8_t tensor_arena[kTensorArenaSize];

  // Declare the task handles
  TaskHandle_t task_inference_handle = NULL;
  TaskHandle_t task_interrupt_handle = NULL;
  TfLiteStatus invoke_status;
}  // namespace

// Globals extern
int node_index_check = -1;
int target_index[4] = {-1, -1, -1, -1};
int notification = -1;
int fault_index = 0;
constexpr int number_of_inputs = sizeof(labels)/sizeof(labels[0]);
constexpr int number_of_targets = (sizeof(targets)/sizeof(targets[0]));
uint8_t *stack_pointer = NULL; // Global variable to store the stack pointer

void setup() {
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<7> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk||
      resolver.AddConv2D() != kTfLiteOk||
      resolver.AddAveragePool2D() != kTfLiteOk||
      resolver.AddSoftmax() != kTfLiteOk||
      resolver.AddTanh() != kTfLiteOk||
      resolver.AddLogistic()!= kTfLiteOk||
      resolver.AddReshape() != kTfLiteOk) {
    MicroPrintf("Operation adding failed");
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  
  // Generate or recover golden from Flash
  save_load_golden(fi_method, interpreter);

  if(fi_method == 4){

    create_timer();

  }
  
  MicroPrintf("\nSetup complete.\n==============");
}

void loop() {
  // Run inference and calculate accuracy for all the samples
  int start_image = 0;

  InferenceTaskArgs* args = new InferenceTaskArgs();
  // Randomize the samples for RegisterFI due to frequent crashes
  if(fi_method == 4) {
    start_image = esp_random() % number_of_inputs;
    //args->interpreter = interpreter;
    //args->mainTaskHandle = xTaskGetCurrentTaskHandle();
  }

  for (int i = start_image; i < number_of_inputs; ++i) {
    // Copy data to input tensor
    for (int l = 0; l < input_size; l++) {
      input->data.raw[l] = image_bin[l+(input_size*i)];
    }

    // MemoryFI
    if(fi_method == 2){
      
      invoke_status = MemoryFI_RAM(i, interpreter);

      if (invoke_status == kTfLiteOk) { 
        analyze_results(i, output->data.int8);
      }
      else{
        vTaskDelay(pdMS_TO_TICKS(100));
      }
    }

    // RegisterFI
    if(fi_method == 4){
      //MicroPrintf("Image: %i\nTarget: %i", i, targets[current_target_index]);
      TfLiteStatus invoke_status = interpreter->Invoke();

      if (invoke_status == kTfLiteOk) { 
        analyze_results(i, output->data.int8);
      }

      current_target_index++;

      if(current_target_index>number_of_targets){
        current_target_index=0;
      }
    }
  }
}




