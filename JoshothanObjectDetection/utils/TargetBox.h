#pragma once

class TargetBox
{
public:
	int x1;
	int x2;
	int y1;
	int y2;

	int cate;
	float score;

	float area() { return getWidth() * getHeight(); };
private:
	float getWidth() { return (x2 - x1); };
	float getHeight() { return (y2 - y1); };
};

