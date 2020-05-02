[![Build Status](https://travis-ci.org/kristopherjohnson/lunar-c.svg?branch=master)](https://travis-ci.org/kristopherjohnson/lunar-c)

LUNAR for C
===========

This is a port of a classic text-based ["lunar lander" game][lunarlander] to [C][c].

The first time I saw a computer was when my father took me to an open-house at the IBM headquarters in Atlanta in the late 1970's, when I was around ten years old.  I wasn't impressed with the big sterile glass rooms filled with big blue computers, but there was a room in the basement where a couple of bearded guys asked me to sit down at a terminal and play a game.

It was this lunar landing game.  For each ten seconds of game time, it asks how much thrust you want to use, and then it tells you your new altitude, velocity, and remaining fuel.  I crashed, and then I had to get up to let the next kid take a turn.

It was simple, primitive even, but I was immediately fascinated with computers.  I saw that a computer would let me create little simulated universes that followed whatever rules I could imagine.

So after that I kept bugging my dad to buy me books about programming.  A couple of years later, my parents bought me a computer.  Thanks Mom and Dad!

I did some research, and I found the [original Lunar Lander program by Jim Storer][storerlunarlander], written in the [FOCAL programming language][wpfocal] in 1969.  I've rewritten it in C so that it can be built and run on modern systems.  The program here is pretty close to what I remember.  One difference is that, on crashing, that program announced "IT'S ALL OVER BUT THE SHOUTING", which confused me as a ten-year-old.  I thought the shouting was for joy.

This code is based upon these sources:

* [Jim Storer's original lunar landing simulation code in FOCAL][storer]
* [FOCAL Programming Manual][focal]
* [David Ahl's port to BASIC in _BASIC Computer Games_ (1978)][ahl]

If you have a UNIX-ish system with development tools installed, you should be able to build the executable and play the game by running this command:

```
make run
```

If you have [Docker][docker], you can run the executable from an existing Docker image without building it:

```
docker run -it --rm oldmankris/lunar
```

Here is an example play session:

```
CONTROL CALLING LUNAR MODULE. MANUAL CONTROL IS NECESSARY
YOU MAY RESET FUEL RATE K EACH 10 SECS TO 0 OR ANY VALUE
BETWEEN 8 & 200 LBS/SEC. YOU'VE 16000 LBS FUEL. ESTIMATED
FREE FALL IMPACT TIME-120 SECS. CAPSULE WEIGHT-32500 LBS


FIRST RADAR CHECK COMING UP


COMMENCE LANDING PROCEDURE
TIME,SECS   ALTITUDE,MILES+FEET   VELOCITY,MPH   FUEL,LBS   FUEL RATE
      0             120      0        3600.00     16000.0      K=:0
     10             109   5016        3636.00     16000.0      K=:0
     20              99   4224        3672.00     16000.0      K=:0
     30              89   2904        3708.00     16000.0      K=:0
     40              79   1056        3744.00     16000.0      K=:0
     50              68   3960        3780.00     16000.0      K=:0
     60              58   1056        3816.00     16000.0      K=:0
     70              47   2904        3852.00     16000.0      K=:170
     80              37   1474        3539.86     14300.0      K=:200
     90              27   5247        3140.80     12300.0      K=:200
    100              19   4537        2710.41     10300.0      K=:200
    110              12   5118        2243.83      8300.0      K=:200
    120               7   2285        1734.97      6300.0      K=:200
    130               3   1990        1176.06      4300.0      K=:200
    140               0   5040         556.96      2300.0      K=:170
    150               0   1040         -21.20       600.0      K=:0
    160               0   1087          14.80       600.0      K=:0
    170               0    606          50.80       600.0      K=:30
    180               0    436         -27.90       300.0      K=:0
    190               0    581           8.10       300.0      K=:8
    200               0    425          13.17       220.0      K=:10
    210               0    253          10.30       120.0      K=:9
    220               0     95          11.11        30.0      K=:100
FUEL OUT AT   220.30 SECS
ON THE MOON AT   226.11 SECS
IMPACT VELOCITY OF    21.35 M.P.H.
FUEL LEFT:     0.00 LBS
CONGRATULATIONS ON A POOR LANDING



TRY AGAIN?
(ANS. YES OR NO):NO
CONTROL OUT


```

[lunarlander]: https://en.wikipedia.org/wiki/Lunar_Lander_(video_game_genre)#Text_games
[c]: https://en.wikipedia.org/wiki/C_(programming_language)
[storerlunarlander]: https://www.cs.brandeis.edu/~storer/LunarLander/LunarLander.html
[wpfocal]: https://en.wikipedia.org/wiki/FOCAL_(programming_language)
[storer]: http://www.cs.brandeis.edu/~storer/LunarLander/LunarLander/LunarLanderListing.jpg
[focal]: http://www.bitsavers.org/www.computer.museum.uq.edu.au/pdf/DEC-08-AJAB-D%20PDP-8-I%20FOCAL%20Programming%20Manual.pdf
[ahl]: https://www.atariarchives.org/basicgames/showpage.php?page=106
[docker]: https://www.docker.com/