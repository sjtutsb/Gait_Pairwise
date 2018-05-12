x=[0 18 36 54 72 90 108 126 144 162 180];
y1=[38.12 53.03 60.17 58.49 55.13 51.57 52.79 53.16 52.91 48.13 37.76];
y2=[36.78 52.21 52.01 50.78 47.16 42.88 44.87 45.68 41.17 39.76 31.50];
y3=[37.13 54.12 58.79 59.68 54.36 49.17 53.14 56.17 53.12 47.89 36.18];
y4=[42.43 57.32 68.17 65.15 56.66 52.49 56.81 61.35 57.14 52.13 41.14];
plot(x,y1,'--o',x,y2,'--d',x,y3,'--s',x,y4','-^','markerfacecolor','k');
axis([0 180 0 100])
set(gca,'XTick',18:18:180);
set(gca,'YTick',0:10:100);
grid on;
title('Gallery: NM $1-4,   view angles: 0��-180��   Probe: CL $1-2');
xlabel('Probe view angle (��)');
ylabel('Identification accuracy(%)');
legend('LB','MT','Siamese','ours');