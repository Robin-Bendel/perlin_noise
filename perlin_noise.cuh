#define FREQUENCY 0.01f
#define OCTAVES 5

double perlin(double x, double y);
double* perlin2DOctave(int octave);
void normalize(double* values);