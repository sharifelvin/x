import time
import datetime as dt
import binance
import config as cfg
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, load, dump
from stockstat import wrap, unwrap


def getusercoins(client):
    temp_coins = client.futures_account_balance()
    user_coins = [x['asset'] for x in temp_coins if (
        float(x['balance']) > 0) and x['asset'] != 'USDT' and x['asset'] != 'SOLO']
    return user_coins


def getcount(client, _market):
    temp_coins = client.futures_account_balance()
    count = [x['balance'] for x in temp_coins
             if (x['asset']) == _market.replace('USDT', '')]
    return count[0]


def getusdtbalance(client):
    temp_coins = client.futures_account_balance()
    user_balance = [float(x['balance'])
                    for x in temp_coins if x['asset'] == 'USDT']
    return user_balance[0]


def gettradablesymbols(client):
    hchange = client.futures_mark_price()
    exclude = ['UP', 'DOWN', 'BEAR', 'BULL', 'BUSD', 'TUSD', 'DAI', 'GBP',
               'SOlO', 'EUR', 'TUSD', 'UST', 'XNO', 'USDC', 'USDP', 'SUSD', 'PAXG']
    symbols = [x['symbol'] for x in hchange if x['symbol'].endswith(
        'USDT') and all(excludes not in x['symbol'] for excludes in exclude)]
    return symbols


def createframe(data):
    df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'Closetime', 'Quoteassetvolume',
                                     'Numberoftrades', 'Takerbuybaseassetvolume', 'Takerbuyquoteassetvolume', 'Ignore'])
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df.open = df.open.astype(float)
    df.high = df.high.astype(float)
    df.low = df.low.astype(float)
    df.close = df.close.astype(float)
    df.volume = df.volume.astype(float)
    df.date = pd.to_datetime(df.date, unit='ms',).dt.tz_localize(
        'UTC').dt.tz_convert('Asia/Baku').dt.tz_localize(None)
    return df


def createdata(client, symbol, frame='15m', new=True, start='2019-01-02 16:00:00+04:00', end=time.strftime("%Y-%m-%d %H:%M:%S")):
    if new:
        data = client.futures_klines(symbol=symbol, interval=frame, limit=1500)
    else:
        data = client.futures_historical_klines(
            symbol=symbol, interval=frame, start_str=start, end_str=end)
    df = wrap(createframe(data))
    return df


def dataloader(symbol):
    try:
        with open('data/'+symbol, 'rb') as f:
            data = load(f)
    except:
        pass
    return data


def sload():
    with open('best', 'rb') as f:
        return load(f)


def symbolsloader(client, update=False):
    if update:
        symbolsdefiner(client)
    try:
        with open('symbols', 'rb') as f:
            symbols = load(f)
    except:
        symbolsdefiner()
        with open('symbols', 'rb') as f:
            symbols = load(f)
    return symbols


def symbolsremover(symbol):
    symbols = symbolsloader()
    symbols.remove(symbol)
    with open('symbols', 'wb') as f:
        dump(symbols, f)


def symbolsdefiner(client, replace=True):
    symbols = gettradablesymbols(client)
    for symbol in symbols:
        df = client.futures_klines(symbol=symbol, interval='15m', limit=10)
        data = wrap(createframe(df))
        t1 = data[-1:].index[0]
        t2 = pd.Timestamp.now()
        if pd.Timedelta(t2 - t1).days > 0:
            print(symbol)
            symbols.remove(symbol)
    if replace:
        with open('symbols', 'wb') as f:
            dump(symbols, f)


def poscleaner(client):
    orders = client.futures_get_all_orders()
    symbols = [order['symbol'] for order in orders if order['status']!='FILLED' and order['status']!='CANCELED'and order['status']!='EXPIRED']
    for symbol in symbols:
        client.futures_cancel_all_open_orders(symbol=symbol)


def getblacklist():
    try:
        return load('blacklist')
    except:
        blacklist = pd.DataFrame(columns=['time'])
        dump(blacklist,'blacklist')
        return blacklist


def getdata(client, symbol, new=True):
    if new:
        df = createdata(client, symbol, '15m', new=True)
        return df
    else:
        try:
            df = dataloader(symbol)[:-2]
            kf = createdata(client, symbol, '15m', new=True)
            df = df.combine_first(kf)
            with open('data/'+symbol, 'wb') as f:
                dump(df, f)
            return df
        except:
            try:
                df = createdata(client, symbol, '15m', new=False)[:-2]
                kf = createdata(client, symbol, '15m', new=True)
                df = df.combine_first(kf)
                with open('data/'+symbol, 'wb') as f:
                    dump(df, f)
                return df
            except:
                try:
                    symbols = symbolsloader()
                    symbols.remove(symbol)
                    with open('symbols', 'wb') as f:
                        dump(symbols, f)
                    return False
                except:
                    return False


def lower(df, ran):
    z, r = 500, 1
    af = pd.DataFrame()
    af.index = df.index
    lows = df[df.ll == 1].index
    df['id'] = list(range(0, len(df)))
    for i in lows[:-1]:
        j = lows[r]
        low1 = df.loc[i].l
        id1 = df.loc[i].id
        id2 = df.loc[j].id
        low2 = df.loc[j].l
        df['lr'] = np.where((df.id > id2+ran) & (df.id < id2+z),
                            low2+((df.id-id2)*((low2-low1)/(id2-id1))), np.nan)
        af['slr'+str(r)] = np.where((df['lr'] < df.open) &
                                    (df['lr'] > df.close) &
                                    (df['lr'] < df.close*1.001),
                                    df['lr'], np.nan)
        af['plr'+str(r)] = np.where((df['lr'] < df.open) &
                                    (df['lr'] > df.close),
                                    df['lr'], np.nan)
        df = df.drop(['lr'], axis=1)
        af = af.copy()
        r += 1
    plrs = [i for i in af.columns if i.startswith('plr')]
    slrs = [i for i in af.columns if i.startswith('slr')]
    af['eps'] = af[plrs].min(axis=1)
    af['short'] = af[plrs].ge(af.eps*0.9995, axis=0).sum(axis=1)
    af['sell'] = af[slrs].ge(af.eps*0.9995, axis=0).sum(axis=1)
    return af['sell'], af['short'], af['eps']


def blower(zf, ran,ref):
    z = 500
    print(ref)
    sho = 0.995
    kont = len(zf)//1000
    if kont == 0:
        kont = 1
    rf = pd.DataFrame()
    for j in range(0, kont):
        df = zf[j*1000:(j+1)*1000+z].copy()
        af = pd.DataFrame()
        af.index = df.index
        mf = af.copy()
        lows = df[df.ll == 1].index
        k, r = 1, 1
        df.loc[:, 'id'] = list(range(0, len(df)))
        for i in lows[:-1]:
            if 1 > 0:
                j = lows[k]
                low1 = df.loc[i].l
                id1 = df.loc[i].id
                id2 = df.loc[j].id
                low2 = df.loc[j].l
                df['lr'] = np.where((df.id > id2+ran) & (df.id < id2+z),
                                    low2+((df.id-id2)*((low2-low1)/(id2-id1))), np.nan)
                af['potlr'+str(r)] = np.where((df['lr'].shift(1) < df.open.shift(1)) &
                                              (df['lr'].shift(1) > df.close.shift(1)), df['lr'].shift(1), np.nan)
                df = df.drop(['lr'], axis=1)
                r += 1
                af = af.copy()
            k += 1
        potlrs = [i for i in af.columns if i.startswith('potlr')]
        mf['poteps'] = af[potlrs].max(axis=1)
        mf['pots'] = np.where(mf.poteps>0,1,0)
        rf = rf.combine_first(mf)
    return rf['poteps'],rf['pots']


def higher(df, ran):
    z, r = 500, 1
    stopdown = 0.991
    af = pd.DataFrame()
    af.index = df.index
    lows = df[df.hh == 1].index
    k, r = 1, 1
    df['id'] = list(range(0, len(df)))
    for i in lows[: -1]:
        j = lows[r]
        low1 = df.loc[i].h
        id1 = df.loc[i].id
        id2 = df.loc[j].id
        low2 = df.loc[j].h
        df['hr'] = np.where((df.id > id2+ran) & (df.id < id2+z),
                            low2+((df.id-id2)*((low2-low1)/(id2-id1))), np.nan)
        af['pothr'+str(r)] = np.where((df['hr'] > df.open), df['hr'], np.nan)
        df = df.drop(['hr'], axis=1)
        af = af.copy()
        r += 1
    pothrs = [i for i in af.columns if i.startswith('pothr')]
    af['potepl'] = af[pothrs].min(axis=1)
    af['potl'] = np.where(af.potepl*stopdown<df.open,1,0)
    return af['buy'], af['long'], af['epl']


def bhigher(zf, ran,line):
    z = 50
    kont = len(zf)//1000
    if kont == 0:
        kont = 1
    rf = pd.DataFrame()
    for j in range(0, kont):
        df = zf[j*1000:(j+1)*1000+z].copy()
        ref = line[j*1000:(j+1)*1000+z].copy()
        af = pd.DataFrame()
        af.index = df.index
        mf = af.copy()
        lows = df[(df.hh == 1)|df.ll==1].index
        k, r = 1, 1
        df.loc[:, 'id'] = list(range(0, len(df)))
        ref.loc[:, 'id'] = list(range(0, len(df)))
        for i in lows:
            id2 = df.loc[i].id
            if len(df[i:])>500:
                i2 = df[df.id ==df.loc[i].id+500].index[0]
            else:
                i2 = df[-1:].index[0]
            df.loc[i:i2,'hr'] = ref[i:i2].open.pct_change().cumsum()
            df.loc[i:i2,'zr'] = df[i:i2].open.pct_change().cumsum()
            df['hr'] = np.where((df.id > id2+ran) & (df.id < id2+z),df.hr, np.nan)
            df['zr'] = np.where((df.id > id2+ran) & (df.id < id2+z),df.zr, np.nan)
            af['pothr'+str(r)] = np.where((df['hr'] > df.zr.shift(1)) &
                                            (df['hr'] < df.zr), 1,0)
            af['potlr'+str(r)] = np.where((df['hr'] < df.zr.shift(1)) &
                                            (df['hr'] > df.zr), 1,0)
            df = df.drop(['hr','zr'], axis=1)
            r += 1
            af = af.copy()
            k += 1
        pothrs = [i for i in af.columns if i.startswith('pothr')]
        potlrs = [i for i in af.columns if i.startswith('potlr')]
        mf['potl'] = af[pothrs].gt(0).sum(axis=1)
        mf['pots'] = af[potlrs].gt(0).sum(axis=1)
        # mf[pothrs] = af[pothrs]
        # mf[potlrs] = af[potlrs]
        # mf['potl'] = np.where(mf.potepl>0,1,0)
        rf = rf.combine_first(mf)
    return rf['potl'],rf['pots']


def pre_decision(client, symbol, ran=3, limit=1, new=True):
    df = getdata(client, symbol, new)
    df = df[-1*(limit+510):]
    df['h'] = np.where(df.open > df.close, df.open, df.close)
    df['l'] = np.where(df.open < df.close, df.open, df.close)
    df['ll'] = np.where((df.l.rolling(2*ran+1, center=True).min()
                         == df.l) & (df.l < df.l.shift(1)) , 1, 0)
    df['hh'] = np.where((df.h.rolling(2*ran+1, center=True).max()
                         == df.h) & (df.h > df.h.shift(1)) , 1, 0)
    df['sl'] = np.nan
    df['sh'] = np.nan
    df['pots'] = blower(df, ran)
    df['potl'] = bhigher(df, ran)
    df['ep'] = df.open
    df['symbol'] = symbol
    df = df.iloc[-1]
    df = df[['symbol','potl','pots','ep']]
    return df


def decisioner(client, symbol,ran=3, limit=1000, new=False):
    z = 100
    leverage = 20
    rang = range(0, z+1)
    df = getdata(client, symbol, new)
    ref = getdata(client, 'BTCUSDT',new)
    stopup = 1.0019
    stopdown = 0.981
    lo = 1.041
    sho = 0.959
    df = df[-1*(limit+500):]
    ref = ref[-len(df):]
    # df['h'] = df.high
    # df['l'] = df.low
    # df['top'] = np.where(df.open > df.close, df.open, df.close)
    # df['bottom'] = np.where(df.open < df.close, df.open, df.close)
    # df['ll'] = np.where((df.l.rolling(2*ran+1, center=True).min()
    #                      == df.l) & (df.l < df.l.shift(1)) &
    #                     (df.bottom.rolling(2*ran+1, center=True).min()
    #                      == df.bottom), 1, 0)
    # df['hh'] = np.where((df.h.rolling(2*ran+1, center=True).max()
    #                      == df.h) & (df.h > df.h.shift(1)) &
    #                     (df.top.rolling(2*ran+1, center=True).max()
    #                      == df.top), 1, 0)
    df['h'] = np.where(df.open > df.close, df.open, df.close)
    df['l'] = np.where(df.open < df.close, df.open, df.close)
    df['ll'] = np.where((df.l.rolling(2*ran+1, center=True).min()
                         == df.l) & (df.l < df.l.shift(1)) , 1, 0)
    df['hh'] = np.where((df.h.rolling(2*ran+1, center=True).max()
                         == df.h) & (df.h > df.h.shift(1)) , 1, 0)
    df['sl'] = np.nan
    df['sh'] = np.nan
    df['potl'],df['pots'] = bhigher(df, ran,ref)
    df['dec'] = df.potl-df.pots
    # df['eps'],df['pots'] = blower(df, ran,ref)
    # df['potl'] = bhigher(df, ran,ref)
    df['bth']=df.open*lo
    df['btl']=df.open*stopdown
    df['sth']=df.open*stopup
    df['stl']=df.open*sho
    df = df[500:]
    # cond = [((df.high.shift(-i) > df.bth) &
    #          (df.low.shift(-i).rolling(i, closed='left').min() > df.btl)) for i in rang]
    # cond.append((df.low.shift(-z).rolling(z, closed='left').min() > df.btl))
    # val = [1+leverage*(((df.bth-df.open)/df.open)-0.001) for i in rang]
    # val.append(1+leverage*(((df.close.shift(-z)-df.open)/df.open)-0.001))
    # values = [i for i in rang]
    # values.append(z-1)
    # df['lpro'] = np.select(cond, val, 1+leverage *
    #                        (((df.btl-df.open)/df.open)-0.001))
    # df['bc'] = np.select(cond, values, 0)
    # cond = [((df.low.shift(-i) < df.stl) &
    #          (df.high.shift(-i).rolling(i, closed='left').max() < df.sth)) for i in rang]
    # cond.append((df.high.shift(-z).rolling(z, closed='left').max() < df.sth))
    # val = [1+leverage*(((df.open-df.stl)/df.open)-0.001) for i in rang]
    # val.append(1+leverage*(((df.open-df.close.shift(-z))/df.open)-0.001))
    # values = [i for i in rang]
    # values.append(z)
    # df['spro'] = np.select(cond, val, 1+leverage *
    #                        (((df.open-df.sth)/df.open)-0.001))
    # df['sc'] = np.select(cond, values, 0)
    df['prof'] = np.select([df.dec>1,df.dec<-1],[
        1+leverage*(((df.close-df.open)/df.open)-0.001),
        1+leverage*(((df.open-df.close)/df.open)-0.001)
        ],1)
    # df = df[:-z]
    df['prof'] = np.where(df.prof<0.8,0.8,df.prof)
    df['symbol'] = symbol
    # df = df[['symbol', 'bc','sc','lpro','spro','potl','pots','dec','prof']]
    df = df[['symbol','potl','pots','dec','prof']]
    return df


def get_signal(client, ran=3, limit=1):
    pd.set_option('display.max_columns', None)
    ran = 3
    limit = 1
    import warnings
    warnings.catch_warnings()
    warnings.simplefilter("ignore")
    xb = sload()
    entry = pd.DataFrame(columns=xb.index)
    swlow = pd.DataFrame(columns=xb.index)
    swhigh = pd.DataFrame(columns=xb.index)
    decision = pd.DataFrame(columns=xb.index)
    data = Parallel(n_jobs=-1)(delayed(pre_decision)(client, symbol,
                                                     ran, limit) for symbol in xb.index)
    time = data[0].name
    data = pd.DataFrame.from_dict(data)
    live = pd.DataFrame().from_dict(data.live).T
    live.columns = data.symbol
    decision = pd.DataFrame().from_dict(data.decision).T
    decision.columns = data.symbol
    entry = pd.DataFrame().from_dict(data.ep).T
    entry.columns = data.symbol
    swlow = pd.DataFrame().from_dict(data.tl).T
    swlow.columns = data.symbol
    swhigh = pd.DataFrame().from_dict(data.th).T
    swhigh.columns = data.symbol
    market = 'nan'
    market = np.select([
        (live.gt(0).sum(axis=1)[0] > 0) & (decision.gt(0).sum(
            axis=1)[0] - decision.lt(0).sum(axis=1)[0] >= 30),
        (live.lt(0).sum(axis=1)[0] > 0) & (decision.lt(0).sum(
            axis=1)[0] - decision.gt(0).sum(axis=1)[0] >= 30)
    ],
        [live.astype(float).idxmax(axis=1), live.astype(float).idxmin(axis=1)], 'nan')[0]
    dec, enpr, sl, sh = 0, 0, 0, 0
    # print(live.gt(0).sum(axis=1)[0],live.lt(0).sum(axis=1)[0],decision.gt(0).sum(axis=1)[0],decision.lt(0).sum(axis=1)[0])
    if ((market != 'nan')):
        dec = live[market][0]
        enpr = entry[market][0]
        sl = swlow[market][0]
        sh = swhigh[market][0]
    return market, dec, enpr, sl, sh, time


def backtest(client, ran=3, limit=1000, new=False):
    pd.set_option('display.max_columns', None)
    import warnings
    warnings.catch_warnings()
    warnings.simplefilter("ignore")
    xb = sload()
    decision = decisioner(client, 'BTCUSDT', ran, limit=limit, new=new)[[]]
    # bc = pd.DataFrame(index=decision.index)
    # sc = pd.DataFrame(index=decision.index)
    # lpro = pd.DataFrame(index=decision.index)
    prof = pd.DataFrame(index=decision.index)
    # spro = pd.DataFrame(index=decision.index)
    potl = pd.DataFrame(index=decision.index)
    pots = pd.DataFrame(index=decision.index)
    data = Parallel(n_jobs=-1)(delayed(decisioner)(client, symbol,
                                                   ran, limit, new=new) for symbol in xb.index)
    for af in data:
        decision[str.upper(af['symbol'].iloc[0])] = af['dec']
        # lpro[str.upper(af['symbol'].iloc[0])] = af['lpro']
        prof[str.upper(af['symbol'].iloc[0])] = af['prof']
        # spro[str.upper(af['symbol'].iloc[0])] = af['spro']
        # bc[str.upper(af['symbol'].iloc[0])] = af['bc']
        # sc[str.upper(af['symbol'].iloc[0])] = af['sc']
        potl[str.upper(af['symbol'].iloc[0])] = af['potl']
        pots[str.upper(af['symbol'].iloc[0])] = af['pots']
    # decision,prof,bc, sc, lpro, spro, potl,pots = decision[xb.index],prof[xb.index],bc[xb.index], sc[xb.index], lpro[xb.index], spro[xb.index], potl[xb.index], pots[xb.index]
    decision,prof,potl,pots = decision[xb.index],prof[xb.index],potl[xb.index], pots[xb.index]
    # lpro.columns = lpro.columns.str.upper()
    prof.columns = prof.columns.str.upper()
    decision.columns = decision.columns.str.upper()
    # spro.columns = spro.columns.str.upper()
    # bc.columns = bc.columns.str.upper()
    # sc.columns = sc.columns.str.upper()
    potl.columns = potl.columns.str.upper()
    pots.columns = pots.columns.str.upper()
    # decision['potl'] = potl.gt(0).sum(axis=1)
    # decision['pots'] = pots.gt(0).sum(axis=1)
    # decision['dec'] = np.select([
    #     (potl.gt(0).sum(axis=1) - pots.gt(0).sum(axis=1) >= 30),
    #     (pots.gt(0).sum(axis=1) - potl.gt(0).sum(axis=1) >= 30)
    # ],
    #     [1, -1], 0)
    decision[decision.columns] = np.where(abs(decision) > 0, decision, 0)
    decision['symbol'] = np.select([((decision.gt(0).sum(axis=1)) - decision.lt(0).sum(axis=1) >= 50), ((
        decision.lt(0).sum(axis=1)) - (decision.gt(0).sum(axis=1)) >= 50)], [decision.idxmax(axis=1), decision.idxmin(axis=1)], 'nan')

    # decision['symbol'] = np.select([decision['dec']==1,
    #                                 decision['dec']==-1],
    #                                 [
    #                                     potl.idxmax(axis=1),
    #                                     pots.idxmax(axis=1)
    #                                 ],'nan')
    k = 0
    # decision['prof'] = np.nan
    # decision['cprof'] = np.where(decision.dec!=0,spro['BTCUSDT'],1)
    for i in decision.index:
        if k == 0:
            if ((str(decision.loc[i, 'symbol']) != 'nan')):
                decision.loc[i, 'dec'] = decision.loc[i,(decision.loc[i, 'symbol'])]
                decision.loc[i, 'prof'] = prof.loc[i,str.upper(decision.loc[i, 'symbol'])]

                # if decision.loc[i, 'dec']>=1:
                #     decision.loc[i, 'prof'] = lpro.loc[i,str.upper(decision.loc[i, 'symbol'])]
                #     decision.loc[i, 'aprof'] = spro.loc[i,str.upper(decision.loc[i, 'symbol'])]
                #     # k = bc.loc[i, str.upper(decision.loc[i, 'symbol'])]
                # elif decision.loc[i, 'dec']<=-1:
                #     decision.loc[i, 'prof'] = spro.loc[i,str.upper(decision.loc[i, 'symbol'])]
                #     decision.loc[i, 'aprof'] = lpro.loc[i,str.upper(decision.loc[i, 'symbol'])]
                    # k = sc.loc[i, str.upper(decision.loc[i, 'symbol'])]
        else:
            k -= 1
    start = 10
    init = start
    tr,fl = 0,0
    ti = 0
    if len(decision[decision.prof > 1]) > 0:
        ti = (decision[decision.prof > 1].index[-1] -
              decision[decision.prof > 1].index[0]).round(freq='1D').days
        for i in decision[(decision.prof > 0)]['prof']:
            if init < 2:
                print('exhaust')
            init *= i
            if i>1:
                tr += 1
            else:
                fl += 1
    print(ran, '| We earned ', init-start, '$ in ', ti, 'day with ',
          tr, ' winning,', fl, ' losing trades', sep='')
    return decision,prof


def calibrate(client, ran=3, limit=1000):
    try:
        with open('next_calibration_time', 'rb') as f:
            next_calibration_time = load(f)
    except:
        next_calibration_time = dt.datetime.now() - dt.timedelta(hours=50)
    if dt.datetime.now() < next_calibration_time:
        print('Already calibrated recently')
        return 1
    pd.set_option('display.max_columns', None)
    import warnings
    warnings.catch_warnings()
    warnings.simplefilter("ignore")
    decision = decisioner(client, 'BTCUSDT', ran, limit=limit)[[]]
    real = pd.DataFrame(index=decision.index)
    skip = pd.DataFrame(index=decision.index)
    symbols = symbolsloader(update=False)
    higlo = pd.DataFrame(columns=['T', 'F', 'P'])
    opeco = pd.DataFrame(columns=['T', 'F', 'P'])
    i = len(symbols)
    for symbol in symbols:
        hl = decisioner(client, symbol, ran, limit)
        oc = decisioner(client, symbol, ran, limit, hl=False)
        t1, f1 = len(hl[(abs(hl.decision) > 0) & (hl.real > 1)]), len(
            hl[(abs(hl.decision) > 0) & (hl.real < 1)])
        t2, f2 = len(oc[(abs(oc.decision) > 0) & (oc.real > 1)]), len(
            oc[(abs(oc.decision) > 0) & (oc.real < 1)])
        higlo.loc[symbol] = [t1, f1, 0]
        opeco.loc[symbol] = [t2, f2, 0]
        opeco['P'] = np.where(
            opeco['F'] > 0, opeco['T']/(opeco['F']), opeco['T']*2)
        higlo['P'] = np.where(
            higlo['F'] > 0, higlo['T']/(higlo['F']), higlo['T']*2)
        if higlo.loc[symbol, 'P'] > opeco.loc[symbol, 'P']:
            af = hl
        else:
            af = oc
        decision[symbol] = af['decision']
        real[symbol] = af['real']
        skip[symbol] = af['skip']
        i -= 1
        print(i, end='\r')
    opeco, higlo = opeco.sort_index(), higlo.sort_index()
    best = pd.DataFrame()
    best.index = higlo.index
    best['type'] = np.where((higlo['P'] > opeco['P']) | ((higlo['P'] == opeco['P']) & (
        (higlo['F'] < opeco['F']) | (higlo['T'] > opeco['T']))), True, False)
    best['point'] = np.where((higlo['P'] > opeco['P']) | ((higlo['P'] == opeco['P']) & (
        (higlo['F'] < opeco['F']) | (higlo['T'] > opeco['T']))), higlo['P'], opeco['P'])
    xb = best.sort_values('point', ascending=False)
    real, decision = real[xb.index], decision[xb.index]
    decision[decision.columns] = np.where(abs(decision) > 0, decision, 0)
    decision['col'] = np.select([(decision.mean(axis=1) > 0), (decision.mean(
        axis=1) < 0)], [decision.idxmax(axis=1), decision.idxmin(axis=1)], 'nan')
    # real['col'] = decision['col']
    real['col'] = 'SOLUSDT'
    k = 0
    for i in decision.index:
        if k == 0:
            if ((str(real.loc[i, 'col']) != 'nan')):
                decision.loc[i, 'dec'] = decision.loc[i,
                                                      (decision.loc[i, 'col'])]
                decision.loc[i, 'prof'] = real.loc[i,
                                                   str.upper(real.loc[i, 'col'])]
                if skip.loc[i, str.upper(real.loc[i, 'col'])] > 0:
                    k = skip.loc[i, str.upper(real.loc[i, 'col'])]
                else:
                    k = 0
        else:
            k -= 1
    start = 150
    init = start
    ti = (decision[decision.prof > 1].index[-1] -
          decision[decision.prof > 1].index[0]).round(freq='1D').days
    for i in decision[decision.prof > 0]['prof']:
        init *= i
    print(ran, '| We earned ', init-start, '$ in ', ti, 'day with ',
          len(decision[decision.prof > 1]), ' winning,', len(decision[decision.prof < 1]), ' losing ', len(decision[(decision.prof == 1)]), ' neutral trades', sep='')
    with open('best', 'wb') as f:
        dump(xb, f)
    next_calibration_time = dt.datetime.now() + dt.timedelta(hours=20)
    with open('next_calibration_time', 'wb') as f:
        dump(next_calibration_time, f)
    print('Calibrated Succesfully')


def init_client():
    client = binance.Client(api_key=cfg.getPublicKey(),
                            api_secret=cfg.getPrivateKey())
    return client


def get_futures_balance(client, _asset="USDT"):
    balances = client.futures_account_balance()
    asset_balance = 0
    for balance in balances:
        if balance['asset'] == _asset:
            asset_balance = balance['balance']
            break
    return float(asset_balance)


def initialise_futures(client, _market="BTCUSDT", _leverage=20, _margin_type="ISOLATED"):
    try:
        client.futures_change_leverage(symbol=_market, leverage=_leverage)
    except Exception as e:
        print(e)

    try:
        client.futures_change_margin_type(
            symbol=_market, marginType=_margin_type)
    except Exception as e:
        return e


def set_exit_time(ti):
    with open('exittime', 'wb') as f:
        dump(ti, f)


def reset_exit_time():
    set_exit_time(False)


def get_exit_time():
    with open('exittime', 'rb') as f:
        return load(f)


def get_orders(client, _market="BTCUSDT"):
    orders = client.futures_get_open_orders(symbol=_market)
    return orders, len(orders)


def get_positions(client):
    positions = client.futures_position_information()
    return positions


def is_in_position(client):
    positions = get_positions(client)
    for position in positions:
        if float(position['positionAmt']) != 0:
            return position['symbol']
    return False


def get_specific_positon(client, _market="BTCUSDT"):
    position = client.futures_position_information(symbol=_market)
    return position[0]


def close_position(client, _market="BTCUSDT"):
    position = get_specific_positon(client, _market)
    qty = float(position['positionAmt'])

    _side = "BUY"
    if qty > 0.0:
        _side = "SELL"

    if qty < 0.0:
        qty = qty * -1

    qty = str(qty)
    client.futures_create_order(symbol=_market, type='MARKET',
                                quantity=qty,
                                side=_side)
    while check_in_position(client=client, _market=_market):
        time.sleep(2)
    client.futures_cancel_all_open_orders(symbol=_market)
    reset_exit_time()
    print(f"Exited Position: {qty} ${_market}")


def get_entry(client, _market="BTCUSDT"):
    position = get_specific_positon(client, _market)
    price = position['entryPrice']
    return price


def check_in_position(client, _market="BTCUSDT"):
    position = get_specific_positon(client, _market)
    in_position = False
    if float(position['positionAmt']) != 0.0:
        in_position = True
    return in_position


def get_market_price(client, _market="BTCUSDT"):
    price = client.futures_ticker(symbol=_market)['lastPrice']
    return float(price)


def handle_signal(client, _market="BTCUSDT", _leverage=20, _order_side="BUY",
                  _entryprice=0, _swinglow=0, _swinghigh=0, _type='LIMIT'):
    client.futures_cancel_all_open_orders(symbol=_market)
    initialise_futures(client, _market=_market, _leverage=_leverage)
    _qty, _entryprice = calculate_position(
        client, _market=_market, _price=_entryprice, _leverage=_leverage)
    client.futures_create_order(symbol=_market,
                                type=_type,
                                side=_order_side,
                                price=_entryprice,
                                timeInForce='GTC',
                                quantity=_qty)
    k = 0
    while k < 60 and not is_in_position(client):
        time.sleep(2)
        k += 1
    if not is_in_position(client) and len(client.futures_get_open_orders(symbol=_market)) > 0:
        client.futures_cancel_all_open_orders(symbol=_market)
        return 0
    else:
        if _order_side == "BUY":
            _stoprice = _swinglow
            _triggerprice = _swinghigh
            _stop_side = "SELL"
        else:
            _stoprice = _swinghigh
            _triggerprice = _swinglow
            _stop_side = "BUY"
        print(
            f"{_order_side}: {_qty} ${_entryprice} using x{_leverage} leverage")

        # log_trade(_qty=qty, _market=market, _leverage=leverage, _side=side,
        #           _cause="Signal", _trigger_price=_entryprice,
        #           _market_price=market_price, _type=order_side)

        # Create a SL/TP order.
        _stoprice = price_pre(client, _market, _stoprice)
        _qty = qty_pre(client, _market, _qty)
        _triggerprice = price_pre(client, _market, _triggerprice)
        if is_in_position(client):
            try:
                client.futures_create_order(symbol=_market,
                                            type='STOP_MARKET',
                                            side=_stop_side,
                                            timeInForce='GTC',
                                            stopPrice=_stoprice,
                                            quantity=_qty)
                client.futures_create_order(symbol=_market,
                                            type='TAKE_PROFIT_MARKET',
                                            side=_stop_side,
                                            timeInForce='GTC',
                                            stopPrice=_triggerprice,
                                            quantity=_qty)
            except:
                close_position(client=client, _market=_market)
                time.sleep(3)
                client.futures_cancel_all_open_orders(symbol=_market)
                
    return 1


def calculate_position(client, _market="BTCUSDT", _price=1, _leverage=1):
    usdt = get_futures_balance(client, _asset="USDT")*.99
    spend = usdt*_leverage
    _qty = float(spend/_price)*0.99
    quantityPrecision = [i['quantityPrecision'] for i in client.futures_exchange_info(
    )['symbols'] if i['symbol'] == _market][0]-1
    pricePrecision = [i['pricePrecision'] for i in client.futures_exchange_info(
    )['symbols'] if i['symbol'] == _market][0]-1
    if pricePrecision == -1:
        pricePrecision = 0
    if quantityPrecision == -1:
        quantityPrecision = 0
    _price = float("{:0.0{}f}".format(_price, pricePrecision))
    _qty = float("{:0.0{}f}".format(_qty, quantityPrecision))
    return _qty, _price


def price_pre(client, _market="BTCUSDT", _price=1):
    pricePrecision = [i['pricePrecision'] for i in client.futures_exchange_info(
    )['symbols'] if i['symbol'] == _market][0]-1
    if pricePrecision == -1:
        pricePrecision = 0
    _price = float("{:0.0{}f}".format(_price, pricePrecision))
    return _price


def qty_pre(client, _market="BTCUSDT", _qty=1):
    quantityPrecision = [i['quantityPrecision'] for i in client.futures_exchange_info(
    )['symbols'] if i['symbol'] == _market][0]-1
    if quantityPrecision == -1:
        quantityPrecision = 0
    _qty = float("{:0.0{}f}".format(_qty, quantityPrecision))
    return _qty


def log_trade(_qty=0, _market="BTCUSDT", _leverage=1, _side="long", _cause="signal", _trigger_price=0, _market_price=0, _type="exit"):
    df = pd.read_csv("trade_log.csv")
    df2 = pd.DataFrame()
    df2['time'] = [time.time()]
    df2['market'] = [_market]
    df2['qty'] = [_qty]
    df2['leverage'] = [_leverage]
    df2['cause'] = [_cause]
    df2['side'] = [_side]
    df2['trigger_price'] = [_trigger_price]
    df2['market_price'] = [_market_price]
    df2['type'] = [_type]
    df = df.append(df2, ignore_index=True)
    df.to_csv("trade_log.csv", index=False)
