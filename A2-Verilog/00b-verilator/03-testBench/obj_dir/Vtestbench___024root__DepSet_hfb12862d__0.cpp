// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vtestbench.h for the primary calling header

#include "Vtestbench__pch.h"
#include "Vtestbench__Syms.h"
#include "Vtestbench___024root.h"

VL_INLINE_OPT VlCoroutine Vtestbench___024root___eval_initial__TOP__Vtiming__0(Vtestbench___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtestbench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtestbench___024root___eval_initial__TOP__Vtiming__0\n"); );
    auto &vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSymsp->TOP____024unit.__VmonitorNum = 1U;
    vlSelfRef.testbench__DOT__a = 0U;
    vlSelfRef.testbench__DOT__b = 0U;
    co_await vlSelfRef.__VdlySched.delay(0x2710ULL, 
                                         nullptr, "testbench.v", 
                                         19);
    vlSelfRef.testbench__DOT__a = 0U;
    vlSelfRef.testbench__DOT__b = 1U;
    co_await vlSelfRef.__VdlySched.delay(0x2710ULL, 
                                         nullptr, "testbench.v", 
                                         20);
    vlSelfRef.testbench__DOT__a = 1U;
    vlSelfRef.testbench__DOT__b = 0U;
    co_await vlSelfRef.__VdlySched.delay(0x2710ULL, 
                                         nullptr, "testbench.v", 
                                         21);
    vlSelfRef.testbench__DOT__a = 1U;
    vlSelfRef.testbench__DOT__b = 1U;
    co_await vlSelfRef.__VdlySched.delay(0x2710ULL, 
                                         nullptr, "testbench.v", 
                                         22);
    VL_FINISH_MT("testbench.v", 24, "");
}

#ifdef VL_DEBUG
VL_ATTR_COLD void Vtestbench___024root___dump_triggers__act(Vtestbench___024root* vlSelf);
#endif  // VL_DEBUG

void Vtestbench___024root___eval_triggers__act(Vtestbench___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtestbench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtestbench___024root___eval_triggers__act\n"); );
    auto &vlSelfRef = std::ref(*vlSelf).get();
    // Body
    vlSelfRef.__VactTriggered.set(0U, ((((IData)(vlSelfRef.testbench__DOT__a) 
                                         != (IData)(vlSelfRef.__Vtrigprevexpr___TOP__testbench__DOT__a__0)) 
                                        | ((IData)(vlSelfRef.testbench__DOT__b) 
                                           != (IData)(vlSelfRef.__Vtrigprevexpr___TOP__testbench__DOT__b__0))) 
                                       | ((IData)(vlSelfRef.testbench__DOT__y) 
                                          != (IData)(vlSelfRef.__Vtrigprevexpr___TOP__testbench__DOT__y__0))));
    vlSelfRef.__VactTriggered.set(1U, vlSelfRef.__VdlySched.awaitingCurrentTime());
    vlSelfRef.__Vtrigprevexpr___TOP__testbench__DOT__a__0 
        = vlSelfRef.testbench__DOT__a;
    vlSelfRef.__Vtrigprevexpr___TOP__testbench__DOT__b__0 
        = vlSelfRef.testbench__DOT__b;
    vlSelfRef.__Vtrigprevexpr___TOP__testbench__DOT__y__0 
        = vlSelfRef.testbench__DOT__y;
    if (VL_UNLIKELY((1U & (~ (IData)(vlSelfRef.__VactDidInit))))) {
        vlSelfRef.__VactDidInit = 1U;
        vlSelfRef.__VactTriggered.set(0U, 1U);
    }
#ifdef VL_DEBUG
    if (VL_UNLIKELY(vlSymsp->_vm_contextp__->debug())) {
        Vtestbench___024root___dump_triggers__act(vlSelf);
    }
#endif
}

VL_INLINE_OPT void Vtestbench___024root___nba_sequent__TOP__0(Vtestbench___024root* vlSelf) {
    (void)vlSelf;  // Prevent unused variable warning
    Vtestbench__Syms* const __restrict vlSymsp VL_ATTR_UNUSED = vlSelf->vlSymsp;
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vtestbench___024root___nba_sequent__TOP__0\n"); );
    auto &vlSelfRef = std::ref(*vlSelf).get();
    // Body
    if (VL_UNLIKELY(((~ (IData)(vlSymsp->TOP____024unit.__VmonitorOff)) 
                     & (1U == vlSymsp->TOP____024unit.__VmonitorNum)))) {
        VL_WRITEF_NX("Time: %0t | a: %b, b: %b, y: %b\n",0,
                     64,VL_TIME_UNITED_Q(1000),-9,1,
                     (IData)(vlSelfRef.testbench__DOT__a),
                     1,vlSelfRef.testbench__DOT__b,
                     1,(IData)(vlSelfRef.testbench__DOT__y));
    }
}
