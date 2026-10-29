#ifndef MODEL
#define MODEL

const int number_of_parameter_tensors = 5;
const int number_of_outputs = 10;
extern unsigned char g_model[];
const int parameter_tensor_positions_lookup[number_of_parameter_tensors][3] = {
{3, 92444, 840},
{5, 51764, 40320},
{7, 3260, 48000},
{9, 772, 2400},
{11, 572, 150},
};

#endif  // MODEL