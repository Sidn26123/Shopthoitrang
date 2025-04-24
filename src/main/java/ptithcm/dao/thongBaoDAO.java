package ptithcm.dao;

import java.util.List;

import ptithcm.entity.ThongBaoEntity;

public interface thongBaoDAO{
	
	public List<ThongBaoEntity> layThongBaoCuaUser(int mand);
	public ThongBaoEntity LayThongBaoTheoMaTB(int matb);
	
	public void addThongBao(ThongBaoEntity thongbao);
	
	public void deleteThongBao(ThongBaoEntity thongbao);
	
	public void updateThongBao(ThongBaoEntity thongBao);
	
	public void markAllNotificationRead(int mand);
}