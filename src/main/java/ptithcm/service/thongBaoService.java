package ptithcm.service;

import java.util.List;

import ptithcm.entity.ThongBaoEntity;

public interface thongBaoService{
	public List<ThongBaoEntity> LayThongBaoCuaUser(int mand);
	public Integer LayTongThongBaoChuaDoc(List<ThongBaoEntity> listTB);
	
	public void addThongBao(ThongBaoEntity thongBao);
	public void deleteThongBao(ThongBaoEntity thongbao);
	public void updateThongBao(ThongBaoEntity thongBao);
	public void markAllNotificationRead(int mand);
	
	public int getNumNotificationUnread(List<ThongBaoEntity> listTB);
}