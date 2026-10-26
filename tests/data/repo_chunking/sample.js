function multiply(a, b) {
    return a * b;
}

function divide(a, b) {
    return b !== 0 ? a / b : 0;
}

function greet(name) {
    return `Hello, ${name}!`;
}

class Calculator {
    constructor() {
        this.history = [];
    }
    
    add(a, b) {
        const result = a + b;
        this.history.push(`${a} + ${b} = ${result}`);
        return result;
    }
}
