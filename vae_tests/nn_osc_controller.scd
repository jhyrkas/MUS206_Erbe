s.boot;

// --------- OSC --------

(
var toFloatArray, toWavetable, recvFunc, recvErrFunc, parameter_func, generate_func, window, ks, generate_b, random_b, size = 32;

// from karl
toFloatArray = {
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
toWavetable = {
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

// receiving a message
recvFunc = OSCFunc({
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
    "received wavetable".postln;
    //b = Buffer.alloc(s, 512);
    //b.loadCollection(array);
}, '/scRecv', nil, 7771);

recvErrFunc = OSCFunc({
	| msg, time, addr, recvPort |
	'error generating wavetable'.postln;
}, '/scErr', nil, 7771);

// sending parameter values
parameter_func = {
    | val_array |
    b = NetAddr.new("127.0.0.1", 1337);
    16.do {
        | index |
        b.sendMsg("/param" ++ index, val_array[index].value);
    };
};

// sending a message to generate the wavetable
generate_func = {
    b = NetAddr.new("127.0.0.1", 1337);
    b.sendMsg("/generate");
};

//fb.value();

// GUI
// dims: x-axis position, y-axis position, x-axis length, y-axis length
window = Window.new("Knobs", Rect(640,730,270,270)).front;
ks = Array.new(16);
4.do {
	| index_i |
	4.do {
		| index_j |
		var tmp_k = Knob.new(window, Rect(10 + 40*index_j, 10 + 40*index_i, size, size));
        tmp_k.centered = true;
        tmp_k.value = 0.5;
		ks.add(tmp_k);
	};
};

generate_b = Button.new(window,Rect(5,210,80,30)).states_([["Generate"]]);
generate_b.action_({
	| butt |
    parameter_func.value(ks);
    generate_func.value();
});

random_b = Button.new(window,Rect(85,210,80,30)).states_([["Randomize"]]);
random_b.action_({
	| butt |
    16.do { |index| ks[index].value = 1.0.rand}});
)

// ---------synthesis---------

// when it's time to play, mess with these parameters
(
{VOsc.ar(SinOsc.ar(0.25, 0.0, 1.99, 2.0), 220 + SinOsc.ar(110, 0.0, 10.0), 0.0, 0.3)}.play;
)

(
{Osc.ar(4, 220, 0.0, 0.3)}.play;
)

// synth def
(
SynthDef.new("nn_wavetable", {
    |carr_freq = 440, voice_center = 0, voice_sweep = 0, voice_sweep_freq = 0.25, gain = 0.3, duration = 2.0, mod_mult = 1.5, fm_index = 0.0 |
    var env, mod_wave, voice_wave, carr_wave, stereo_mix;
    // attack time, release time, peak level
    env = Env.perc(0.1, duration, 0.3).kr(doneAction: 2);
    // frequency modulation wave
    mod_wave = SinOsc.ar(freq: carr_freq * mod_mult, mul: fm_index);
	// voice_modulation
	voice_wave = SinOsc.ar(freq: voice_sweep_freq, mul: voice_sweep, add: voice_center);
    // carrier wave
	carr_wave = VOsc.ar(bufpos: voice_wave, freq: carr_freq, mul: gain);
    // still need filter here
    // stereo mix
    stereo_mix = Splay.ar(carr_wave);
    Out.ar(0, stereo_mix);}).add;
)

(
var notes, durs, note_stream, dur_stream;

notes = Pseq([Pseq([82, 84, 87, 91] - 24, 4), Pseq([84, 87, 91, 94] - 24, 2), Pseq([84, 85, 89, 92] - 24, 2)], 4);
durs = Pseq([0.25], 32*4);

note_stream = notes.asStream();
dur_stream = durs.asStream();

r = Task({
    var dur, note, i = 0;
    while ({
        i < 63; // this was supposed to be 64 but i kind of like it at 63
    }, {
        dur = dur_stream.next;
        note = note_stream.next.midicps;
        Synth("nn_wavetable",
			[carr_freq: note, duration: dur, voice_center: 4.0 * (100.rand / 100.0)]);
        dur.yield;
        i = i + 1;
    });

    while ({
        i < 128;
    }, {
        dur = dur_stream.next;
        note = note_stream.next.midicps;
        Synth("nn_wavetable",
            [carr_freq: note, fm_index: 190.linrand + 10, duration: dur, mod_mult: 0.5 + 1.5.rand,
			 voice_center: 3.0* (100.rand / 100.0), voice_sweep: 1.0, voice_sweep_freq: 0.25 + 4.75.rand
		]);
        dur.yield;
        i = i + 1;
    });
}).play(quant: TempoClock.default.beats); // don't think quantization is working like i want it to
)