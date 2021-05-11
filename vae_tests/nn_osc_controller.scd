s.boot;

(
var fa;
fa = {
    | param = 0, val = 0 |
    b = NetAddr.new("127.0.0.1", 1337);
    b.sendMsg("/param" ++ param, val);
    [param, val];
};

fa.value(4,1.3);
)

(
var fb;
fb = {
    b = NetAddr.new("127.0.0.1", 1337);
    b.sendMsg("/generate");
    0
};

fb.value();
)