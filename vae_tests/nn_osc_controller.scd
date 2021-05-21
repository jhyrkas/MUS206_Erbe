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

16.do {
    | index |
    fa.value(index, 2.0.rand - 1.0);
};
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
// strong assumption that we are receiving 512 arrays
var toWavetable = {
    | array |
    var return = FloatArray.newClear(1024);
    512.do {
        | index |
        var i = index * 2;
        var index2 = (index + 1) % 512;
        return[i] = 2*array[index] - array[index2];
        return[i+1] = array[index2] - array[index];
    };
    return
};

var recvFunc = OSCFunc({
	| msg, time, addr, recvPort |
    var array, wavetable;
    array = toFloatArray.(msg[1]);
    wavetable = toWavetable.(array);
    //s.sendMsg(\b_alloc, i, 512);

    //s.performList(\sendMsg, \b_gen, i, \sine1, 7, a);
    // the argument '7' here is a flag for the \sine1 wave fill method
    // TODO: need to figure out which flag to set?
    b = Buffer.alloc(s, 1024);
    b.loadCollection(wavetable);
    //b = Buffer.alloc(s, 512);
    //b.loadCollection(array);
}, '/scRecv', nil, 7771);
)

// when it's time to play, mess with these parameters
(
{VOsc.ar(SinOsc.ar(0.25, 0.0, 0.99, 1.0), 220, 0.0, 0.3)}.play;
)

(
{Osc.ar(2, 220, 0.0, 0.3)}.play;
)