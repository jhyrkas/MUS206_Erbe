s.boot;

// sending parameter values
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

// receiving a message
(
var recv_n;
var recvFunc;

//recv_n = NetAddr.new("127.0.0.1", 7771); // receiving here
recvFunc = OSCFunc({
	| msg, time, addr, recvPort |
	[msg, time, addr, recvPort].postln;
    msg.size.postln;
    msg.at(1).size.postln;
}, '/scRecv', nil, 7771);
)

// -------

(
var fb;
fb = {
    b = NetAddr.new("127.0.0.1", 1337);
    b.sendMsg("/generate");
    0
};

fb.value();
)

b = [1,2,3];
b.size
b.at(0)