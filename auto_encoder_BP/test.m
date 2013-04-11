close all;
clear;
x0 = load('out0123_0');
x1 = load('out0123_1');
x2 = load('out0123_2');
x3 = load('out0123_3');

plot(x0(:,1),x0(:,2),'rx');
hold on;
plot(x1(:,1),x1(:,2),'bo');
plot(x2(:,1),x2(:,2),'k+');
plot(x3(:,1),x3(:,2),'g*');
legend({'class 0', 'class 1','class 2','class 3'});

