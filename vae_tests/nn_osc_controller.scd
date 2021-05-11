s.boot;

(
var fa;
fa = {
    | p0 = 0, p1 = 0 |
    b = NetAddr.new("127.0.0.1", 1337);
    b.sendMsg("/param0", p0);
    b.sendMsg("/param1", p1);
    [p0, p1];
};

fa.value(1,0);
)