package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

type Person struct {
    Name string
    Age  int
}

func (p Person) String() string {
    return fmt.Sprintf("%s is %d years old", p.Name, p.Age)
}
