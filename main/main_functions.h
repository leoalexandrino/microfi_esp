#ifndef MAIN_FUNCTIONS
#define MAIN_FUNCTIONS

extern int node_index_check;
extern int fault_index;
extern int fi_method;

extern int targets[5];
extern int target_index[4];

extern uint8_t *stack_pointer;
extern int notification;


// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

// Initializes all data needed for the example. The name is important, and needs
// to be setup() for Arduino compatibility.
void setup();

// Runs one iteration of data gathering and inference. This should be called
// repeatedly from the application code. The name needs to be loop() for Arduino
// compatibility.
void loop();

#ifdef __cplusplus
}
#endif

#endif  // MAIN_FUNCTIONS
