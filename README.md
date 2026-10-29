# MicroFI 0.4

Fault Injection Framework for TensorFlow Lite for Microcontrollers

Leonardo Alexandrino de Melo

Lyon Institute of Nanotechnology

## Abstract

Evaluating the reliability of deep neural network (DNN) applications is essential to assess efficient error resilience techniques and deploy artificial intelligence (AI) on safety-critical
embedded systems. This work presents a novel methodology for assessing the reliability of DNN models deployed on microcontrollers using software-level fault injection. A new tool was developed, on top of the TensorFlow Lite library for C/C++, to inject faults into memory and CPU registers during DNN inference. Providing a portable, fast, and configurable approach for injecting multiple
fault types. The proposed framework was validated with two case studies, using a variety of literature-endorsed models and datasets, the results provide a comparative analysis of DNN reliability versus register and memory bit flips, induced by transient faults.

## Paper:
A. de Melo, A. Bosio, R. P. Bastos, "MicroFI: TensorFlow Lite based Fault Injection Framework for Microcontrollers," IEEE VLSI Test Symposium (VTS 2025), Tempe, United States, Apr 2025

## Notes

Use notebook to generate models and datasets that can be imported to the microcontroller on C. Tested on ESP32C3. TFLite/LiteRT was modified to achieve fast fault injection, you need to use the patched library available on this repo. 