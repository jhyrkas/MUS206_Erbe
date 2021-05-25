// based on https://doc.sccode.org/Guides/GUI-Introduction.html
(
w = Window.new("GUI Introduction", Rect(200,200,255,100));
b = Button.new(w,Rect(10,0,80,30)).states_([["Generate"],["Add to bank"]]);
s = Slider.new(w,Rect(95,0,150,30));
c = CompositeView.new(w,Rect(20,35,100,60));
StaticText.new(c,Rect(0,0,80,30)).string_("Hello");
StaticText.new(c,Rect(20,30,80,30)).string_("World!");
b.action = { c.visible = b.value.asBoolean.not };
s.action = { c.bounds = Rect( s.value * 150 + 20, 35, 100, 100 ) };
w.front;
)

(
var window, ks, generate_b, size = 32;
// dims: x-axis position, y-axis position, x-axis length, y-axis length
window = Window.new("Knobs", Rect(640,730,270,270)).front;
ks = Array.new(16);
4.do {
	| index_i |
	4.do {
		| index_j |
		var tmp_k = Knob.new(window, Rect(10 + 40*index_j, 10 + 40*index_i, size, size));
		ks.add(tmp_k);
	};
};

generate_b = Button.new(window,Rect(5,210,80,30)).states_([["Generate"]]);
generate_b.action_({
	| butt |
	var arr = Array.new(16);
	16.do { |index| arr.add(ks[index].value);};
	arr.postln;
});
)