#ifndef DATA_H
#define DATA_H
#include <pgmspace.h>

const float x_train_data[3][3] PROGMEM = {
  {0.0f, 0.0f, 0.0f},
  {1.0f, 1.0f, 1.0f},
  {1.0f, 0.0f, 0.0f}
};

const float y_train_data[3][2] PROGMEM = {
  {1.0f, 0.0f},
  {0.0f, 1.0f},
  {0.0f, 0.0f}
};

#endif // DATA_H