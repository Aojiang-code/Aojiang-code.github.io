
/*@preserve
***Version 2.10.0***
*/

/*@license
 *                       Copyright 2002 - 2018 Qualtrics, LLC.
 *                                All rights reserved.
 *
 * Notice: All code, text, concepts, and other information herein (collectively, the
 * "Materials") are the sole property of Qualtrics, LLC, except to the extent
 * otherwise indicated. The Materials are proprietary to Qualtrics and are protected
 * under all applicable laws, including copyright, patent (as applicable), trade
 * secret, and contract law. Disclosure or reproduction of any Materials is strictly
 * prohibited without the express prior written consent of an authorized signatory
 * of Qualtrics. For disclosure requests, please contact notice@qualtrics.com.
 */

try {
  (window["WAFQualtricsWebpackJsonP-cloud-2.10.0"]=window["WAFQualtricsWebpackJsonP-cloud-2.10.0"]||[]).push([[7],{39:function(e,n,t){"use strict";t.r(n);var d=function(e,n){this.payload=n,this.type=e};t.d(n,"addPopunderEmbeddedDataHandler",(function(){return o})),t.d(n,"setEmbeddedData",(function(){return a})),t.d(n,"updatePopunderEDCallback",(function(){return i})),t.d(n,"persistEDToSurvey",(function(){return s}));var o=function(e){var n=window.QSI,t=n.util,d=n.windowHandler,o=n.dbg;t.observe(window,"beforeunload",e,!0);try{d.setupWindowHandles()}catch(e){o.e(e)}},r=function(e,n){if("string"==typeof e&&"string"==typeof n){var t=window.QSI.windowHandler,o=t.getWin(n);if(o){var r=new d("setTargetUrlInPlaceholderWindow",e),a=JSON.stringify(r),i=t.getWindowOrigin(window);o.postMessage(a,i)}}},a=function(e,n,t,o){if(void 0!==e&&void 0!==n&&"string"==typeof t&&"string"==typeof o){var r=window.QSI,a=r.windowHandler,i=r.dbg,s=a.getWin(t);if(s){var w=new d("setEmbeddedData",{key:e,value:n}),c=JSON.stringify(w);if(/targetwindow/.test(t))s.postMessage(c,o);else try{var l=s.document.getElementById("PopUnderTargetFrame");if(l&&l.contentWindow)l.contentWindow.postMessage(c,o)}catch(e){i.e(e)}}}},i=function(){var e=window.QSI.dbg;try{var n=window.QSI.windowHandler;n.removeClosedWindowHandles();var t=n.getOptInIDsAndWindowNames()||{},d=n.getOptInIDsAndTargetOrigins()||{};for(var o in t)if(Object.prototype.hasOwnProperty.call(t,o)){var i=t[o],s=d[o]||"*",w=window.QSI.EmbeddedData,c=w.getEmbeddedData(o);if(!c||window.QSI.util.isObjectEmpty(c))continue;if(window.QSI.reg[o]&&/placeholderWindow/.test(i)){var l=window.QSI.reg[o].getTarget();return void r(l,i)}var u=w.surveyOpenED[o];for(var g in c)!Object.prototype.hasOwnProperty.call(c,g)||null!=u&&u[g]==c[g]||a(g,c[g],i,s)}}catch(n){e.e(n)}},s=function(e,n,t){var o=window.QSI,r=o.dbg,a=o.global,i=new d("setEmbeddedData",{key:n,value:t});try{e instanceof HTMLIFrameElement?e.contentWindow.postMessage(i,a.brandBaseUrl||"/"):e.postMessage(i,a.brandBaseUrl||"/")}catch(e){r.e(e)}}}}]);
} catch(e) {
  if (typeof QSI !== 'undefined' && QSI.dbg && QSI.dbg.e) {
    QSI.dbg.e(e);
  }
}