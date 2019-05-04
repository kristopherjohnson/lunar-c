// Translation of
// <http://www.cs.brandeis.edu/~storer/LunarLander/LunarLander/LunarLanderListing.jpg>
// to C.
//
// goto labels and functions with names like "_01_20" or "_06_10" refer to lines
// in the original FOCAL code.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Global variables
//
// A - Altitude (miles)
// G - Gravity
// I - Intermediate altitude (miles)
// J - Intermediate velocity (sec)
// K - Fuel rate (lbs/sec)
// L - Elapsed time (sec)
// M - Total weight (lbs)
// N - Empty weight (lbs, Note: M - N is remaining fuel weight)
// S - Time elapsed in current 10-second turn (sec)
// T - Time remaining in current 10-second turn (sec)
// V - Downward speed (miles/sec)
// W - Temporary working variable
// Z - Thrust per pound of fuel burned

static double A, G, I, J, K, L, M, N, S, T, V, W, Z;

static int _echo_input = 0;

static void _06_10();
static void _09_10();

static int _accept_double(double *value);
static int _accept_yes_or_no();
static void _accept_line(char **buffer, size_t *buffer_length);

int main(int argc, const char **argv)
{
    if (argc > 1)
    {
        // If --echo is present, then write all input back to standard output.
        // (This is useful for testing with files as redirected input.)
        if (strcmp(argv[1], "--echo") == 0)
            _echo_input = 1;
    }

    puts("CONTROL CALLING LUNAR MODULE. MANUAL CONTROL IS NECESSARY");
    puts("YOU MAY RESET FUEL RATE K EACH 10 SECS TO 0 OR ANY VALUE");
    puts("BETWEEN 8 & 200 LBS/SEC. YOU'VE 16000 LBS FUEL. ESTIMATED");
    puts("FREE FALL IMPACT TIME-120 SECS. CAPSULE WEIGHT-32500 LBS\n\n");

_01_20:
    puts("FIRST RADAR CHECK COMING UP\n\n");
    puts("COMMENCE LANDING PROCEDURE");
    puts("TIME,SECS   ALTITUDE,MILES+FEET   VELOCITY,MPH   FUEL,LBS   FUEL RATE");

    A = 120;
    V = 1;
    M = 32500;
    N = 16500;
    G = .001;
    Z = 1.8;
    L = 0;

_02_10:
    printf("%7.0f%16.0f%7.0f%15.2f%12.1f      ",
           L,
           trunc(A),
           5280 * (A - trunc(A)),
           3600 * V,
           M - N);
prompt:
    fputs("K=:", stdout);
    int input = _accept_double(&K);
    if (input != 1 || K < 0 || ((0 < K) && (K < 8)) || K > 200)
    {
        fputs("NOT POSSIBLE", stdout);
        for (int x = 1; x <= 51; ++x)
            putchar('.');
        goto prompt;
    }

    T = 10;

_03_10:
    if (M - N < .001)
        goto _04_10;

    if (T < .001)
        goto _02_10;

    S = T;

    if (N + S * K - M > 0)
        S = (M - N) / K;

    _09_10();

    if (I <= 0)
        goto _07_10;

    if (V <= 0)
        goto _03_80;

    if (J < 0)
        goto _08_10;

_03_80:
    _06_10();

    goto _03_10;

_04_10:
    printf("FUEL OUT AT %8.2f SECS\n", L);
    S = (sqrt(V * V + 2 * A * G) - V) / G;
    V += G * S;
    L += S;

_05_10:
    printf("ON THE MOON AT %8.2f SECS\n", L);
    W = 3600 * V;
    printf("IMPACT VELOCITY OF %8.2f M.P.H.\n", W);
    printf("FUEL LEFT: %8.2f LBS\n", M - N);
    if (W < 1)
        puts("PERFECT LANDING !-(LUCKY)");
    else if (W < 10)
        puts("GOOD LANDING-(COULD BE BETTER)");
    else if (W < 22)
        puts("CONGRATULATIONS ON A POOR LANDING");
    else if (W < 40)
        puts("CRAFT DAMAGE. GOOD LUCK");
    else if (W < 60)
        puts("CRASH LANDING-YOU'VE 5 HRS OXYGEN");
    else
    {
        puts("SORRY,BUT THERE WERE NO SURVIVORS-YOU BLEW IT!");
        printf("IN FACT YOU BLASTED A NEW LUNAR CRATER %8.2f FT. DEEP\n", W * .277777);
    }

    puts("\n\n\nTRY AGAIN?");
    if (_accept_yes_or_no())
        goto _01_20;
    else
    {
        puts("CONTROL OUT\n\n");
        exit(0);
    }

_07_10:
    if (S < .005)
        goto _05_10;
    S = 2 * A / (V + sqrt(V * V + 2 * A * (G - Z * K / M)));
    _09_10();
    _06_10();
    goto _07_10;

_08_10:
    W = (1 - M * G / (Z * K)) / 2;
    S = M * V / (Z * K * (W + sqrt(W * W + V / Z))) + 0.5;
    _09_10();
    if (I <= 0)
        goto _07_10;
    _06_10();
    if (-J < 0)
        goto _03_10;
    if (V <= 0)
        goto _03_10;
    goto _08_10;

    return 0;
}

void _06_10()
{
    L += S;
    T -= S;
    M -= S * K;
    A = I;
    V = J;
}

void _09_10()
{
    double Q = S * K / M;
    double Q_2 = pow(Q, 2);
    double Q_3 = pow(Q, 3);
    double Q_4 = pow(Q, 4);
    double Q_5 = pow(Q, 5);

    J = V + G * S + Z * (-Q - Q_2 / 2 - Q_3 / 3 - Q_4 / 4 - Q_5 / 5);
    I = A - G * S * S / 2 - V * S + Z * S * (Q / 2 + Q_2 / 6 + Q_3 / 12 + Q_4 / 20 + Q_5 / 30);
}

// Read a floating-point value from stdin.
//
// Returns 1 on success, or 0 if input did not contain a number.
//
// Calls exit(-1) on EOF or other failure to read input.
int _accept_double(double *value)
{
    char *buffer = NULL;
    size_t buffer_length = 80;
    _accept_line(&buffer, &buffer_length);
    int input = sscanf(buffer, "%lf", value);
    free(buffer);
    return input;
}

// Reads input and returns 1 if it starts with 'Y' or 'y', or returns 0 if it
// starts with 'N' or 'n'.
//
// If input starts with none of those characters, prompts again.
//
// If unable to read input, calls exit(-1);
int _accept_yes_or_no()
{
prompt:
    fputs("(ANS. YES OR NO):", stdout);
    char *buffer = NULL;
    size_t buffer_length = 80;
    _accept_line(&buffer, &buffer_length);

    if (buffer_length > 0)
    {
        switch (buffer[0])
        {
        case 'y':
        case 'Y':
            free(buffer);
            return 1;
        case 'n':
        case 'N':
            free(buffer);
            return 0;
        default:
            break;
        }
    }
    free(buffer);
    goto prompt;
}

// Reads a line of input.  Caller is responsible for calling free() on the
// returned buffer.
//
// If unable to read input, calls exit(-1).
void _accept_line(char **buffer, size_t *buffer_length)
{
    if (getline(buffer, buffer_length, stdin) == -1)
    {
        fputs("\nEND OF INPUT\n", stderr);
        exit(-1);
    }

    if (_echo_input)
    {
        fputs(*buffer, stdout);
    }
}
