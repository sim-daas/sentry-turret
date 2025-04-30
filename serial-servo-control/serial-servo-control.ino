#include <Servo.h>

Servo panSrv;
Servo tltSrv;

int curPanPos = 90;
int curTltPos = 90;
int tgtPanPos = 90;
int tgtTltPos = 90;
int laserEn = 0;

unsigned long lstMvTm;
unsigned long lstCmdTm;
const int MVTHRESH = 3;
const int SRVMIN = 10;
const int SRVMAX = 170;
const int CMDTMO = 100;
const int LASERPIN = 7;

const int UPDINTV = 20;
const float MAXSPD = 6.0;
float Kp = 0.25;
const float MINANGCHG = 2.0;

String inBuf = "";
boolean cmdCmplt = false;

void setup()
{
  Serial.begin(9600);
  panSrv.attach(9);
  tltSrv.attach(6);
  pinMode(LASERPIN, OUTPUT);
  digitalWrite(LASERPIN, LOW);

  panSrv.write(curPanPos);
  tltSrv.write(curTltPos);

  lstMvTm = millis();
  lstCmdTm = millis();

  delay(500);
}

void loop()
{
  while (Serial.available() > 0)
  {
    char inChr = (char)Serial.read();

    if (isDigit(inChr) || inChr == ',')
    {
      inBuf += inChr;
      lstCmdTm = millis();
    }
    else if (inChr == '\n' || inChr == '\r')
    {
      cmdCmplt = true;
    }
  }

  if (cmdCmplt ||
      (inBuf.length() > 0 && millis() - lstCmdTm > CMDTMO))
  {
    if (inBuf.length() > 0)
    {
      int comIdx1 = inBuf.indexOf(',');

      if (comIdx1 > 0)
      {
        int comIdx2 = inBuf.indexOf(',', comIdx1 + 1);

        if (comIdx2 > 0)
        {
          // Format: pan,tilt,laser
          String panStr = inBuf.substring(0, comIdx1);
          String tltStr = inBuf.substring(comIdx1 + 1, comIdx2);
          String laserStr = inBuf.substring(comIdx2 + 1);

          int newPanTgt = panStr.toInt();
          int newTltTgt = tltStr.toInt();
          int newLaser = laserStr.toInt();

          newPanTgt = constrain(newPanTgt, SRVMIN, SRVMAX);
          newTltTgt = constrain(newTltTgt, SRVMIN, SRVMAX);
          newLaser = constrain(newLaser, 0, 1);

          if (abs(newPanTgt - tgtPanPos) >= MINANGCHG)
          {
            tgtPanPos = newPanTgt;
          }

          if (abs(newTltTgt - tgtTltPos) >= MINANGCHG)
          {
            tgtTltPos = newTltTgt;
          }

          laserEn = newLaser;
          digitalWrite(LASERPIN, laserEn ? HIGH : LOW);
        }
        else
        {
          // Legacy format: pan,tilt
          String panStr = inBuf.substring(0, comIdx1);
          String tltStr = inBuf.substring(comIdx1 + 1);

          int newPanTgt = panStr.toInt();
          int newTltTgt = tltStr.toInt();

          newPanTgt = constrain(newPanTgt, SRVMIN, SRVMAX);
          newTltTgt = constrain(newTltTgt, SRVMIN, SRVMAX);

          if (abs(newPanTgt - tgtPanPos) >= MINANGCHG)
          {
            tgtPanPos = newPanTgt;
          }

          if (abs(newTltTgt - tgtTltPos) >= MINANGCHG)
          {
            tgtTltPos = newTltTgt;
          }
        }
      }
      else
      {
        // Legacy format: pan only
        int newTgt = inBuf.toInt();
        newTgt = constrain(newTgt, SRVMIN, SRVMAX);

        if (abs(newTgt - tgtPanPos) >= MINANGCHG)
        {
          tgtPanPos = newTgt;
        }
      }
    }

    inBuf = "";
    cmdCmplt = false;
  }

  unsigned long curTm = millis();
  if (curTm - lstMvTm >= UPDINTV)
  {
    lstMvTm = curTm;

    updSrvPos(panSrv, tgtPanPos, curPanPos);
    updSrvPos(tltSrv, tgtTltPos, curTltPos);
  }
}

void updSrvPos(Servo &srv, int &tgtPos, int &curPos)
{
  int posErr = tgtPos - curPos;

  if (abs(posErr) > 0)
  {
    float spd = posErr * Kp;

    spd = constrain(spd, -MAXSPD, MAXSPD);

    if (abs(spd) < 0.1)
    {
      curPos = tgtPos;
    }
    else
    {
      curPos += spd;
    }

    curPos = constrain(curPos, SRVMIN, SRVMAX);

    srv.write(round(curPos));
  }
}
