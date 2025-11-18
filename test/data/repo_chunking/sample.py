def fibonacci(n):
    """Calculate the nth Fibonacci number using recursion."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


class MathUtils:
    """Utility class for mathematical operations."""

    def __init__(self):
        self.pi = 3.14159

    def circle_area(self, radius):
        """Calculate the area of a circle."""
        return self.pi * radius**2

    def circle_circumference(self, radius):
        """Calculate the circumference of a circle."""
        return 2 * self.pi * radius


def hello():
    """A simple hello function."""
    return "Hello, World!"
