s.boot;
b = 0; // global buffer num

// sending parameter values
(
var fa;
fa = {
    | param = 0, val = 0 |
    b = NetAddr.new("127.0.0.1", 1337);
    b.sendMsg("/param" ++ param, val);
    [param, val];
};

fa.value(15, 2.0.rand - 1.0);
)

// sending a message to generate the wavetable
(
var fb;
fb = {
    b = NetAddr.new("127.0.0.1", 1337);
    b.sendMsg("/generate");
};

fb.value();
)

// receiving a message
(

// from karl
var toFloatArray = {
    |byte|
    var at = 0;
    var many = byte.size / 4;
    var return = FloatArray.newClear(many);
    many.do {
        |index|
        var int = // this is the order that works. seems little endian
        ((byte[at + 3] & 255) << 24) |
        ((byte[at + 2] & 255) << 16) |
        ((byte[at + 1] & 255) << 8) |
        ((byte[at + 0] & 255) << 0);
        /* ((byte[at + 0] & 255) << 24) |
        ((byte[at + 1] & 255) << 16) |
        ((byte[at + 2] & 255) << 8) |
        ((byte[at + 3] & 255) << 0); // this ordering does not work for numpy arrays */
        at = 4 + at;
        return[index] = Float.from32Bits(int);
    };
    return
};

var recv_n;
var recvFunc;

//recv_n = NetAddr.new("127.0.0.1", 7771); // receiving here
recvFunc = OSCFunc({
	| msg, time, addr, recvPort |
    var array, new_buf;
    array = toFloatArray.(msg[1]);
    //s.sendMsg(\b_alloc, i, 512);

    //s.performList(\sendMsg, \b_gen, i, \sine1, 7, a);
    // the argument '7' here is a flag for the \sine1 wave fill method
    // TODO: need to figure out which flag to set?
    new_buf = Buffer.alloc(s, 512);
    new_buf.loadCollection(array);
}, '/scRecv', nil, 7771);
)

{VOsc.ar(0.2, 220, 0.0, 0.3)}.play;