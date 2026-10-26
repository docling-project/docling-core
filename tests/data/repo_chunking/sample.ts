interface Point {
    x: number;
    y: number;
}

function distance(p1: Point, p2: Point): number {
    return Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
}

class Vector {
    constructor(public x: number, public y: number) {}
    
    magnitude(): number {
        return Math.sqrt(this.x ** 2 + this.y ** 2);
    }
    
    add(other: Vector): Vector {
        return new Vector(this.x + other.x, this.y + other.y);
    }
}
