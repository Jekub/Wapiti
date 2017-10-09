/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.4
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package fr.limsi.wapiti;

public class WapitiIO {
  private long swigCPtr;
  protected boolean swigCMemOwn;

  protected WapitiIO(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(WapitiIO obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        WapitiJNI.delete_WapitiIO(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  protected void swigDirectorDisconnect() {
    swigCMemOwn = false;
    delete();
  }

  public void swigReleaseOwnership() {
    swigCMemOwn = false;
    WapitiJNI.WapitiIO_change_ownership(this, swigCPtr, false);
  }

  public void swigTakeOwnership() {
    swigCMemOwn = true;
    WapitiJNI.WapitiIO_change_ownership(this, swigCPtr, true);
  }

  public String readline() {
    return (getClass() == WapitiIO.class) ? WapitiJNI.WapitiIO_readline(swigCPtr, this) : WapitiJNI.WapitiIO_readlineSwigExplicitWapitiIO(swigCPtr, this);
  }

  public void append(String data) {
    if (getClass() == WapitiIO.class) WapitiJNI.WapitiIO_append(swigCPtr, this, data); else WapitiJNI.WapitiIO_appendSwigExplicitWapitiIO(swigCPtr, this, data);
  }

  public WapitiIO() {
    this(WapitiJNI.new_WapitiIO(), true);
    WapitiJNI.WapitiIO_director_connect(this, swigCPtr, swigCMemOwn, true);
  }

}
