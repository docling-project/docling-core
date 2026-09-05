#include <stdio.h>
#include <math.h>

double circle_area(double radius) {
    return M_PI * radius * radius;
}

double circle_circumference(double radius) {
    return 2 * M_PI * radius;
}

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    printf("Hello, World!\n");
    return 0;
}
