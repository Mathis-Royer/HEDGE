﻿//+------------------------------------------------------------------+
//|                                                         main.mqh |
//|                                       Copyright 2022, Hedge Ltd. |
//|                                            https://www.hedge.com |
//+------------------------------------------------------------------+
#property copyright     "Copyright 2022, Hedge Ltd."
#property link          "https://www.hedge.com"
#property description   "main program"
#property version       "1.00"
//+--------------------------------------------------------------------------------------------------------------+
#include "C:\Users\royer\AppData\Roaming\MetaQuotes\Terminal\24F345EB9F291441AFE537834F9D8A19\MQL5\Include\Hedge_include\market_data.mqh"
#include "C:\Users\royer\AppData\Roaming\MetaQuotes\Terminal\24F345EB9F291441AFE537834F9D8A19\MQL5\Include\Hedge_include\indicators.mqh"
#include "C:\Users\royer\AppData\Roaming\MetaQuotes\Terminal\24F345EB9F291441AFE537834F9D8A19\MQL5\Include\Hedge_include\structure.mqh"
#include "C:\Users\royer\AppData\Roaming\MetaQuotes\Terminal\24F345EB9F291441AFE537834F9D8A19\MQL5\Include\Hedge_include\MetaData.mqh"

//+--------------------------------------------------------------------------------------------------------------+

bool socksend(int sock,string request) 
{
   char req[];
   int  len=StringToCharArray(request,req)-1;
   
   if(len<0) return(false);
   
   return(SocketSend(sock,req,len)==len); 
}

string socketreceive(int sock,int timeout)
{
   char rsp[];
   string result="";
   uint len;
   uint timeout_check=GetTickCount()+timeout;
   do
   {
      len=SocketIsReadable(sock);
      if(len)
      {
         int rsp_len;
         rsp_len=SocketRead(sock,rsp,len,timeout);
         
         if(rsp_len>0) result+=CharArrayToString(rsp,0,rsp_len); 
      }
   }while((GetTickCount()<timeout_check) && !IsStopped());
   
   return result;
}

//+--------------------------------------------------------------------------------------------------------------+

void OnInit()
{
   getAllData();
   int socket=SocketCreate();
   if(socket!=INVALID_HANDLE)
   {
      if(SocketConnect(socket,"127.0.0.1",9090,1000))
      {
         Print("Connected to "," 127.0.0.1",":",9090);
            
         //double clpr[];
         //int copyed = FONCTION_RECUP_DATA(); //Fonction pour récupérer toutes les données d'entrainnement et de fonctionnement du RN : à rédiger
          
         //string tosend;
         //for(int i=0; i<ArraySize(clpr); i++) tosend += (string)clpr[i] + " ";
         
         string received = socksend(socket, "Envoie du programme MQL5") ? socketreceive(socket, 10) : "";     //socketreceive --> b:BUY ; s:SELL ; w:WAIT 
         
         printf("message reçu de la part de python : %s", received);
         //FONCTION_PASSAGE_ORDRE(recieved); //fct° de passage d'ordre : à rédiger
      }
      
      else Print("Connection ","127.0.0.1",":",9090," error ",GetLastError());
      SocketClose(socket);
   }
   else Print("Socket creation error ",GetLastError());
}

//+--------------------------------------------------------------------------------------------------------------+

void getAllData()
{
   //FileDelete("DataMQL5.csv",FILE_COMMON);
   int m_file_handle  = FileOpen("DataMQL5.csv",FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI);
   /*
   printf("existance : %ld", FileGetInteger(m_file_handle,FILE_EXISTS));
   printf("existance 2 : %d", FileIsExist("DataMQL5.txt"));
   printf("last modify date : %s", TimeToString(FileGetInteger(m_file_handle,FILE_MODIFY_DATE),TIME_MINUTES));
   printf("position : %ld", FileGetInteger(m_file_handle,FILE_POSITION));
   printf("is txt : %ld", FileGetInteger(m_file_handle,FILE_IS_TEXT));
   printf("handle : %d", m_file_handle);
   */
   FileWrite(m_file_handle,"time\topen\thigh\tlow\tclose\ttick_volume\tspread\ttick_closeHigh\ttick_closeLow\tvariation_closeOpen\tvariation_closeHigh\tvariation_closeLow\taverage_price\tVolumeProfile\tADX\tADX_PDI\tADX_NDI\tAO\tATR\tBearsPower\tBullsPower\tVar_BBP\tCCI\tDEMA\tVar_DEMA\tTenkan\tKijun\tSBB\tSSA\tVar_Tenkan\tVar_Kijun\tVar_SSB\tVar_SSA\tVar_SSBSSA\tMACD\tMomentum\tRSI\tRVI\tSTOCH\tUO");

   MyMqlRates Myrates[];
   
   int nb_jour=2;
   int jour_init=9;
   
   for(int i=jour_init;i>jour_init-nb_jour;i--)
   {
      //---- get MyMqlRates data
      datetime From = StringToTime("2023.02." + (string)i + " " + "09:00:00");
      printf("time=%s",TimeToString(From,TIME_DATE|TIME_SECONDS));
      datetime time_current = StringToTime("2023.02." + (string)i + " " + "18:00:00");
      int temp0 = getRatesSeconde(_Symbol,From, time_current,Myrates, true, true, true, true, true, true, true, true, true, true, true, true, true);
      
      //---- get Indicators data
      double ADX[],PDI[],NDI[],AO[],ATR[],BearsPower[],BullsPower[],Var_BBP[],CCI[],DEMA[],Var_DEMA[],Tenkan[],Kijun[],SBB_Buffer[],SSA[],Var_Tenkan[],Var_Kijun[],Var_SSB[],Var_SSA[],Var_SSBSSA[],MACD[],Signal_MACD[],Momentum[],RSI[],RVI[],Signal_RVI[],STOCH[],Signal_STOCH[],UO[];
      long VolumeProfile[];
      
      //getVolumeProfile(_Symbol,Myrates,VolumeProfile);
      getADX(14,ADX,PDI,NDI,Myrates);
      getAO(5,34,AO,Myrates);
      getATR(14,ATR,Myrates);
      getBearsBullsPower(13,13,BearsPower,BullsPower,Var_BBP,Myrates);
      getCCI(14,CCI,Myrates);
      getDEMA(14,0,DEMA,Var_DEMA,Myrates);
      getICHIMOKU(9,26,52,Tenkan,Kijun,SBB_Buffer,SSA,Var_Tenkan,Var_Kijun,Var_SSB,Var_SSA,Var_SSBSSA,Myrates);
      getMACD(12,26,9,MACD,Signal_MACD,Myrates);
      getMomentum(14,Momentum,Myrates);
      getRSI(14,RSI,Myrates);
      getRVI(10,3,RVI,Signal_RVI,Myrates);
      getSTOCH(5,3,3,STOCH,Signal_STOCH,Myrates);
      getUO(7,14,28,UO,4,2,1,Myrates);
      
      printf("Array size : ratesSeconde = %d\n", ArraySize(Myrates));
      printf("ADX=%d\n,PDI=%d\n,NDI=%d\n,AO=%d\n,ATR=%d\n,Bears=%d\n,Bulls=%d\n,Var_BBP=%d\n,CCI=%d\n,DEMA=%d\n,Var_DEMA=%d\n,Tenkan=%d\n,Kijun=%d\n,SSB=%d\n,SSA=%d\n,Var_tenkan=%d\n,Var_Kijun=%d\n,Var_SSB=%d\n,Var_SSA=%d\n,Var_SSBSSA=%d\n,MACD=%d\n,Signal_MACD=%d\n,Mom=%d\n,RSI=%d\n,RVI=%d\n,Signal_RVI=%d\n,STOCH=%d\n,Signal_Stoch=%d\n,UO=%d",
              ArraySize(ADX),ArraySize(PDI),ArraySize(NDI),ArraySize(AO),ArraySize(ATR),ArraySize(BearsPower),ArraySize(BullsPower),ArraySize(Var_BBP),ArraySize(CCI),ArraySize(DEMA),ArraySize(Var_DEMA),ArraySize(Tenkan),ArraySize(Kijun),ArraySize(SBB_Buffer),ArraySize(SSA),
              ArraySize(Var_Tenkan),ArraySize(Var_Kijun),ArraySize(Var_SSB),ArraySize(Var_SSA),ArraySize(Var_SSBSSA),ArraySize(MACD),ArraySize(Signal_MACD),ArraySize(Momentum),ArraySize(RSI),ArraySize(RVI),ArraySize(Signal_RVI),ArraySize(STOCH),ArraySize(Signal_STOCH),ArraySize(UO));
      
      //---- write data into csv file
      for(int k=0; k<ArraySize(Myrates)-1-200;k++) FileWrite(m_file_handle,TimeToString(Myrates[k].time,TIME_DATE)+" "+TimeToString(Myrates[k].time,TIME_SECONDS),Myrates[k].open,Myrates[k].high,Myrates[k].low,Myrates[k].close,Myrates[k].tick_volume,Myrates[k].spread,Myrates[k].ticks_closeHigh,Myrates[k].ticks_closeLow,Myrates[k].variation_closeOpen,Myrates[k].variation_closeHigh,Myrates[k].variation_closeLow,Myrates[k].average_price,ADX[k],PDI[k],NDI[k],AO[k],ATR[k],BearsPower[k],BullsPower[k],Var_BBP[k],CCI[k],DEMA[k],Var_DEMA[k],Tenkan[k],Kijun[k],SBB_Buffer[k],SSA[k],Var_Tenkan[k],Var_Kijun[k],Var_SSB[k],Var_SSA[k],Var_SSBSSA[k],MACD[k],Signal_MACD[k],Momentum[k],RSI[k],RVI[k],Signal_RVI[k],STOCH[k],Signal_STOCH[k],UO[k]);
   }
   FileClose(m_file_handle);
   
   
   m_file_handle  = FileOpen("DataMQL5.csv",FILE_COMMON|FILE_READ|FILE_CSV);
   
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));
   printf("read : %s", FileReadString(m_file_handle));

   FileClose(m_file_handle);
   
   return;
}
