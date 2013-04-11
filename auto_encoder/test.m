close all;
plot(x0(:,1),x0(:,2),'rx');
hold on;
plot(x1(:,1),x1(:,2),'bo');
plot(x2(:,1),x2(:,2),'k+');
plot(x3(:,1),x3(:,2),'g*');
legend({'class 0', 'class 1','class 2','class 3'});
